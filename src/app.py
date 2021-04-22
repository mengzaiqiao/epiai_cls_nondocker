#!/usr/bin/python3
from flask import Flask, jsonify, request

from doc_cls import DocCls

app = Flask(__name__)
# tagger = Tagger()
classifier = DocCls()


@app.route("/cls_text")
def cls_text():
    original_text = request.args.get("txt", "", type=str)
    labels, scores = classifier.predict_text(original_text)
    return jsonify(text=[original_text], label=labels, scores=scores)


@app.route("/cls_json", methods=["POST"])
def cls_json():
    print(request.is_json)
    content = request.get_json()
    text_list = content["docs"]
    labels, scores = classifier.predict_text(text_list)
    return jsonify(text=text_list, label=labels, scores=scores)


# @app.route("/link_entity")
# def link_entity():
#     original_text = request.args.get("txt", "", type=str)
#     snomed_name, snomed_id = tagger.normalize(original_text)
#     return jsonify(text=original_text, entities=[f"{snomed_name} ({snomed_id})"])


@app.route("/")
def hello_world():
    json_example_str = ""
    with open("examples/json_example") as f:
        json_example_str = "<br>".join(f.readlines())
    return (
        "Demo 1 (GET+txt), click here : <a href=\"./cls_text?txt=Today%20I%20woke%20up%20with%20migraine%20and%20I%20took%20an%20aspirine.\">./cls_text?txt='Today I woke up with migraine and I took an aspirine.'</a><br>"
        + "Demo 2 (GET+txt), click here: <a href=\"./cls_text?txt=Niger%20states%20have%20reported%20confirmed%20human%20cases%20of%20avian%20influenza%20H5N1,%20also%20called%20bird%20flu.\">./cls_text?txt='Niger states have reported confirmed human cases of avian influenza H5N1, also called bird flu.'</a>"
        + "<br><br><br>"
        + "Demo 3 (POST+json, ), run with command line:<br>"
        + json_example_str
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0")
