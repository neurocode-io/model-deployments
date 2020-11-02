import logging
import json
from onnxruntime import InferenceSession
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import azure.functions as func


dir = Path.cwd()
model_path_list = [str(x) for x in dir.glob("*") if str(x).endswith("model")]
if len(model_path_list) != 1:
    raise RuntimeError("Could not find model")

model_path = model_path_list[0]

fast_tokenizer = AutoTokenizer.from_pretrained(model_path)
session = InferenceSession(f"{model_path}/german-distiled-optimized-quantized.onnx")


def create_error(error_given: str):
    return func.HttpResponse(
        json.dumps({"error": error_given}),
        mimetype="application/json",
        status_code=400,
    )


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    req_query = req.params

    setning = req_query.get("setning")

    if setning is None:
        return create_error("setning missing")

    if setning.count("<mask>") != 1:
        return create_error("either <mask> is missing or more than one <mask>")

    if len(setning) > 512:
        return create_error("Sentence too long")

    result = fill_mask_onnx(setning.replace("<mask>", "[MASK]"))

    return func.HttpResponse(json.dumps(result), mimetype="application/json")


def fill_mask_onnx(setning: str):
    tokens = fast_tokenizer(setning, return_tensors="np")

    if "token_type_ids" in tokens:
        tokens.pop("token_type_ids")

    output = session.run(None, tokens.__dict__["data"])
    token_logits = output[0]

    mask_token_index = np.where(tokens["input_ids"] == fast_tokenizer.mask_token_id)[1]
    mask_token_logits_onnx1 = token_logits[0, mask_token_index, :]

    score = np.exp(mask_token_logits_onnx1) / np.exp(mask_token_logits_onnx1).sum(-1, keepdims=True)

    top_5_idx = (-score[0]).argsort()[:5]
    top_5_values = score[0][top_5_idx]

    result = []

    for token, s in zip(top_5_idx.tolist(), top_5_values.tolist()):
        result.append(f"{setning.replace(fast_tokenizer.mask_token, fast_tokenizer.decode([token]))} (score: {s})")

    return {"result": result}
