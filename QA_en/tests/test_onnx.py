import json
from QA_en.predict import main
import azure.functions as func


def test_happy_path():
    req = func.HttpRequest(
        method="GET", 
        body=None, 
        url="/api/predict", 
        body={
            "context": "Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the `run_squad.py`.", 
            "question": "What is extractive question answering?"
            }
        )

    resp = main(req)
    json_resp = json.loads(resp.get_body())

    assert len(json_resp.get("result")) == 4


def test_question_not_given():
    req = func.HttpRequest(
        method="GET", 
        body=None, 
        url="/api/predict", 
        body={
            "context": "Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the `run_squad.py`.", 
            "bks": "What is extractive question answering?"
            }
        )

    resp = main(req)

    assert resp.status_code == 400


def test_context_not_given():
    req = func.HttpRequest(
        method="GET", 
        body=None, 
        url="/api/predict", 
        body={
            "hkl": "Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the `run_squad.py`.", 
            "question": "What is extractive question answering?"
        }
    )

    resp = main(req)

    assert resp.status_code == 400


def test_too_long_question():
    req = func.HttpRequest(
        method="GET", 
        body=None, 
        url="/api/predict", 
        body={
            "context": "Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the `run_squad.py`.", 
            "question": "We where wondering, on this sunny day, qhat is extractive question answering and what is it used for?"
        }
    )

    resp = main(req)

    assert resp.status_code == 400

