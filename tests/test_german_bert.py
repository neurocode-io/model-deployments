import json
from GermanBert.predict import main
import azure.functions as func


def test_happy_path():
    req = func.HttpRequest(method="GET", body=None, url="/api/predict", params={"setning": "Wie geht es dir <mask>."})

    resp = main(req)
    json_resp = json.loads(resp.get_body())

    assert json_resp == {
        "result": [
            "Wie geht es dir ?. (score: 0.12614840269088745)",
            "Wie geht es dir gut. (score: 0.08869311213493347)",
            "Wie geht es dir besser. (score: 0.0865059345960617)",
            "Wie geht es dir weiter. (score: 0.07658857107162476)",
            "Wie geht es dir geht. (score: 0.028004691004753113)",
        ]
    }
