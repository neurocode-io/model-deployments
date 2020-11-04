import json
from GermanBert.predict import main
import azure.functions as func


def test_happy_path():
    req = func.HttpRequest(method="GET", body=None, url="/api/predict", params={"setning": "Wie geht es dir <mask>."})

    resp = main(req)
    json_resp = json.loads(resp.get_body())

    assert len(json_resp.get("result")) == 5


def test_sentence_not_given():
    req = func.HttpRequest(method="GET", body=None, url="/api/predict", params={"bke": "Wie geht es dir <mask>."})

    resp = main(req)

    assert resp.status_code == 400


def test_sentence_no_mask():
    req = func.HttpRequest(method="GET", body=None, url="/api/predict", params={"sentence": "Wie geht es dir"})

    resp = main(req)

    assert resp.status_code == 400


def test_sentence_multiple_mask():
    req = func.HttpRequest(
        method="GET", body=None, url="/api/predict", params={"sentence": "Wie geht es dir <mask> <mask>"}
    )

    resp = main(req)

    assert resp.status_code == 400


def test_too_long_sentence():
    req = func.HttpRequest(
        method="GET", body=None, url="/api/predict", params={"sentence": "Wie geht es dir <mask>." * 100}
    )

    resp = main(req)

    assert resp.status_code == 400
