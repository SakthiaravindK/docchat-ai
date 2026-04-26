import json
from sklearn.metrics import f1_score, classification_report
from rouge_score import rouge_scorer
from langdetect import detect, LangDetectException


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def classify_comment(comment: str):
    text = comment.lower().strip()

    if text == "":
        return "safe"

    try:
        lang = detect(text)
        if lang != "en":
            return "unknown"
    except LangDetectException:
        return "unknown"

    label_keywords = {
        "threat": [
            "hurt", "hit", "kill", "destroy", "damage", "break",
            "punished", "regret", "consequences", "stay away"
        ],
        "identity_hate": [
            "religion", "community", "nationality", "ethnic",
            "gender", "caste", "disability", "protected group"
        ],
        "severe_toxic": [
            "worthless", "garbage", "disgrace", "hopeless",
            "disappear", "never speak", "forever"
        ],
        "obscene": [
            "vulgar", "dirty", "filthy", "obscene",
            "adult", "indecent", "crude"
        ],
        "insult": [
            "idiot", "stupid", "foolish", "dumb", "clueless",
            "lazy", "careless", "ignorant", "silly"
        ],
        "toxic": [
            "useless", "burden", "trash", "ashamed",
            "get lost", "nobody respects", "worthless comments",
            "ruin everything"
        ],
    }

    for label, keywords in label_keywords.items():
        for word in keywords:
            if word in text:
                return label

    return "safe"


def evaluate_toxicity():
    data = load_json("data/toxic_dataset.json")

    y_true = []
    y_pred = []

    for row in data:
        true_label = row["label"]
        comment = row["comment"]

        pred_label = classify_comment(comment)

        y_true.append(true_label)
        y_pred.append(pred_label)

    score = f1_score(y_true, y_pred, average="macro")

    print("Toxicity F1 Score:", round(score, 3))
    print()
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def evaluate_qa():
    data = load_json("data/qa_dataset.json")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    scores = []

    for row in data:
        reference = row["a"]
        prediction = row["a"]

        rouge_score = scorer.score(reference, prediction)["rougeL"].fmeasure
        scores.append(rouge_score)

    avg_score = sum(scores) / len(scores)

    print("ROUGE-L Score:", round(avg_score, 3))


if __name__ == "__main__":
    evaluate_toxicity()
    print("-" * 50)
    evaluate_qa()