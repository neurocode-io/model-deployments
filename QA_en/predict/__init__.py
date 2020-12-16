import logging
from transformers import AutoTokenizer
from pathlib import Path
import azure.functions as func
from onnx_utils import predict_qa
from onnxruntime import InferenceSession

dir = Path.cwd()
model_path_list = [str(x) for x in dir.glob("*") if str(x).endswith("model")]
if len(model_path_list) != 1:
    raise RuntimeError("Could not find model")

model_path_onnx = model_path_list[0]
squad_model = "deepset/roberta-base-squad2"

fast_tokenizer = AutoTokenizer.from_pretrained(squad_model)

session = InferenceSession(model_path_onnx)

def create_error(error_given: str):
    return func.HttpResponse(
        json.dumps({"error": error_given}),
        mimetype="application/json",
        status_code=400,
    )


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    body = req.get_json()

    if "context" not in body.keys() or "question" not in body.keys():
        return create_error("Question and / or context are missing")

    question = body["question"]
    context = body["context"]

    if question is None or context is None:
        return create_error("Question and / or context are empty")

    examples_dict = {"context": context, "question": question}
    result = predict_qa(model_path_onnx, fast_tokenizer, examples_dict)
