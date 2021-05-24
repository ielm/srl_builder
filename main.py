from allennlp.predictors.predictor import Predictor

import allennlp_models.tagging
import json
import time

from pprint import pprint


def process_dialog(dialog: dict, srl_predictor: Predictor, dep_predictor: Predictor):
    for turn in dialog["turns"]:
        turn["srl"] = extract_srl(turn["text"], srl_predictor)
        turn["deps"] = extract_deps(turn["text"], dep_predictor)
    return dialog


def extract_srl(text: str, srl_predictor):
    return srl_predictor.predict(sentence=text)


def extract_deps(text: str, dep_predictor):
    return dep_predictor.predict(sentence=text)


if __name__ == "__main__":
    with open(
        "/remote/ayuser/leoni/gator_data/Persuasion/svo_dialogs.json",
        "r",
        encoding="utf8",
    ) as file:
        data = [d for d in json.loads(file.read()) if not "BAD" in d["dialog_id"]]

    srl_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
    )
    dep_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
    )

    for dialog in data:
        print(f"\nProcessing dialog: {dialog['dialog_id']}")
        start_time = time.time()
        process_dialog(dialog, srl_predictor=srl_predictor, dep_predictor=dep_predictor)
        print(f"Dialog Execution Time: {time.time()-start_time}")

    with open(
        "/remote/ayuser/leoni/gator_data/Persuasion/srl_dialogs.json",
        "w",
        encoding="utf8",
    ) as wfile:
        wfile.write(json.dumps(data))
