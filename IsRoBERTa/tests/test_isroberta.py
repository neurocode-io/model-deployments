import json
from IsRoBERTa.predict import main
import azure.functions as func


def test_happy_path():
    req = func.HttpRequest(method="GET", body=None, url="/api/predict", params={"setning": "hvernig hefur <mask> það."})

    resp = main(req)
    json_resp = json.loads(resp.get_body())

    assert len(json_resp.get('result')) == 5


def test_sentence_not_given():
    req = func.HttpRequest(method="GET", body=None, url="/api/predict", params={"bke": "hvernig hefur <mask> það."})

    resp = main(req)

    assert resp.status_code == 400

def test_sentence_no_mask():
    req = func.HttpRequest(method="GET", body=None, url="/api/predict", params={"sentence": "hvernig hefur það"})

    resp = main(req)

    assert resp.status_code == 400

def test_sentence_multiple_mask():
    req = func.HttpRequest(method="GET", body=None, url="/api/predict", params={"sentence": "hvernig hefur <mask> það <mask> <mask>"})

    resp = main(req)

    assert resp.status_code == 400

def test_too_long_sentence():
    req = func.HttpRequest(method="GET", body=None, url="/api/predict", params={"sentence": "hvernig hefur <mask> það."*100})

    resp = main(req)

    assert resp.status_code == 400

