from transformers.data.processors import SquadFeatures, squad_convert_examples_to_features
from transformers.pipelines import QuestionAnsweringArgumentHandler
import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy


padding = "longest"
max_seq_len = 384
doc_stride = 128
max_answer_len = 20
max_question_len = 64


def load_tokenizer(pretrained_model="deepset/roberta-base-squad2"):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    return tokenizer


def get_examples(example_dict):
    _arg_parser = QuestionAnsweringArgumentHandler()
    examples = _arg_parser(example_dict)
    return examples


def get_features(examples, tokenizer):
    features_list = []
    if not tokenizer.is_fast:
        features_list = [
            squad_convert_examples_to_features(
                examples=[example],
                tokenizer=self.tokenizer,
                max_seq_length=max_seq_len,
                doc_stride=doc_stride,
                max_query_length=max_question_len,
                padding_strategy=PaddingStrategy.MAX_LENGTH.value,
                is_training=False,
                tqdm_enabled=False,
            ) for example in examples
        ]
    else:
        features_list = []
        for example in examples:
            # Define the side we want to truncate / pad and the text/pair sorting
            question_first = bool(tokenizer.padding_side == "right")
            encoded_inputs = tokenizer(
                text=example.question_text if question_first else example.context_text,
                text_pair=example.context_text if question_first else example.question_text,
                padding=padding,
                truncation="only_second" if question_first else "only_first",
                max_length=max_seq_len,
                stride=doc_stride,
                return_tensors="np",
                return_token_type_ids=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )

            num_spans = len(encoded_inputs["input_ids"])
            p_mask = np.asarray(
                [
                    [tok != 1 if question_first else 0 for tok in encoded_inputs.sequence_ids(
                        span_id)]
                    for span_id in range(num_spans)
                ]
            )

            # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
            if tokenizer.cls_token_id:
                cls_index = np.nonzero(
                    encoded_inputs["input_ids"] == tokenizer.cls_token_id)
                p_mask[cls_index] = 0

            features = []
            for span_idx in range(num_spans):
                features.append(
                    SquadFeatures(
                        input_ids=encoded_inputs["input_ids"][span_idx],
                        attention_mask=encoded_inputs["attention_mask"][span_idx],
                        token_type_ids=encoded_inputs["token_type_ids"][span_idx],
                        p_mask=p_mask[span_idx].tolist(),
                        encoding=encoded_inputs[span_idx],
                        # We don't use the rest of the values - and actually
                        # for Fast tokenizer we could totally avoid using SquadFeatures and SquadExample
                        cls_index=None,
                        token_to_orig_map={},
                        example_index=0,
                        unique_id=0,
                        paragraph_len=0,
                        token_is_max_context=0,
                        tokens=[],
                        start_position=0,
                        end_position=0,
                        is_impossible=False,
                        qas_id=None,
                    )
                )
                features_list.append(features)
    return features_list


def decode(start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int):
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]

    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
    return start, end, candidates[0, start, end]


def predict_qa(model_path, tokenizer, examples_dict):
    if "context" not in examples_dict.keys() or "question" not in examples_dict.keys():
        raise RuntimeError("Wrong keys in the given dictionary")
        
    examples=get_examples(examples_dict)
    features_list=get_features(examples, tokenizer)
    all_answers = []
    for features, example in zip(features_list, examples):
        model_input_names = tokenizer.model_input_names + ["input_ids"]
        fw_args = {k: [feature.__dict__[k] for feature in features]
                   for k in model_input_names}
        session = InferenceSession(model_path)
        output = session.run(None, fw_args)
        start = output[0]
        end = output[1]

        min_null_score = 1000000  # large and positive
        answers = []
        for (feature, start_, end_) in zip(features, start, end):
            # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
            undesired_tokens = np.abs(
                np.array(feature.p_mask) - 1) & feature.attention_mask

            # Generate mask
            undesired_tokens_mask = undesired_tokens == 0.0

            # Make sure non-context indexes in the tensor cannot contribute to the softmax
            start_ = np.where(undesired_tokens_mask, -10000.0, start_)
            end_ = np.where(undesired_tokens_mask, -10000.0, end_)

            # Normalize logits and spans to retrieve the answer
            start_ = np.exp(
                start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
            end_ = np.exp(
                end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))

            # Mask CLS
            start_[0] = end_[0] = 0.0
            starts, ends, scores = decode(
                start=start_, end=end_, topk=1, max_answer_len=max_answer_len)
            if not tokenizer.is_fast:
                char_to_word = np.array(example.char_to_word_offset)
                answers += [
                    {
                        "score": score.item(),
                        "start": np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
                        "end": np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
                        "answer": " ".join(
                            example.doc_tokens[feature.token_to_orig_map[s]                                               : feature.token_to_orig_map[e] + 1]
                        ),
                    }
                    for s, e, score in zip(starts, ends, scores)
                ]
            else:
                question_first = bool(tokenizer.padding_side == "right")
                enc = feature.encoding
                # Sometimes the max probability token is in the middle of a word so:
                # - we start by finding the right word containing the token with `token_to_word`
                # - then we convert this word in a character span with `word_to_chars`
                answers += [
                    {
                        "score": score.item(),
                        "start": enc.word_to_chars(
                            enc.token_to_word(s), sequence_index=1 if question_first else 0)[0],
                        "end": enc.word_to_chars(enc.token_to_word(e), sequence_index=1 if question_first else 0)[1],
                        "answer": example.context_text[
                            enc.word_to_chars(enc.token_to_word(s), sequence_index=1 if question_first else 0)[0]: enc.word_to_chars(enc.token_to_word(e), sequence_index=1 if question_first else 0)[1]],
                    }
                    for s, e, score in zip(starts, ends, scores)
                ]
            answers = sorted(
                answers, key=lambda x: x["score"], reverse=True)[: 1]
            all_answers += answers
        if len(all_answers) == 1:
            return all_answers[0]
        return all_answers
