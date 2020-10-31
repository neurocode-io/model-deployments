import logging
from pathlib import Path
from onnxruntime import InferenceSession
import numpy as np
from transformers import RobertaTokenizerFast
import azure.functions as func

fast_tokenizer = RobertaTokenizerFast.from_pretrained('model/', max_len=512)
session = InferenceSession('model/isroberta-mask.onnx')

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()
    fill_mask_onnx(req_body.get("setning") or "Hann for að <mask>.")
    
    
    return func.HttpResponse(f"Hello, world. This HTTP triggered function executed successfully.")


def fill_mask_onnx(setning):
    tokens = fast_tokenizer(setning, return_tensors="np")
    output = session.run(None,tokens.__dict__['data'])
    token_logits=output[0]

    mask_token_index = np.where(tokens['input_ids'] == fast_tokenizer.mask_token_id)[1]
    mask_token_logits_onnx1 = token_logits[0, mask_token_index, :]

    score = np.exp(mask_token_logits_onnx1) / np.exp(mask_token_logits_onnx1).sum(-1, keepdims=True)  

    top_5_idx = (-score[0]).argsort()[:5]
    top_5_values = score[0][top_5_idx]

    result = []

    for token, s in zip(top_5_idx.tolist(), top_5_values.tolist()):
        result.append(f"{setning.replace(fast_tokenizer.mask_token, fast_tokenizer.decode([token]))} (score: {s*10})")

    return {
        'result': result
    }
