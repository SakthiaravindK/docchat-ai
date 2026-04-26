from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect, LangDetectException

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os


app = FastAPI(title="DocChat AI")


# ---------------------------
# Optional LoRA model loading
# ---------------------------
MODEL_PATH = "models/docchat-lora"
BASE_MODEL = "sshleifer/tiny-gpt2"

tokenizer = None
model = None

try:
    if os.path.exists(MODEL_PATH):
        print("Loading fine-tuned model...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(model, MODEL_PATH)
        model.eval()
        print("Fine-tuned model loaded successfully.")
    else:
        print("LoRA model folder not found. Using rule-based fallback.")
except Exception as e:
    print("Model loading failed:", e)
    print("Using rule-based fallback.")


# ---------------------------
# Request Models
# ---------------------------
class PredictRequest(BaseModel):
    comment: str


class AskRequest(BaseModel):
    document: str
    question: str


class BatchPredictRequest(BaseModel):
    comments: list[str]


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    return {"message": "DocChat AI is running"}


@app.get("/health")
def health():
    return {
        "status": "running",
        "model_loaded": model is not None
    }


# ---------------------------
# Toxic Comment Detection
# ---------------------------
def classify_comment(comment: str):
    text = comment.lower().strip()

    if text == "":
        return "safe", 1.0, "No text to analyse"

    try:
        lang = detect(text)
        if lang != "en":
            return "unknown", 0.0, "Non-English comment"
    except LangDetectException:
        return "unknown", 0.0, "Language detection failed"

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
            "disappear", "never speak", "forever",
            "total disgrace", "endless humiliation",
            "beyond useless", "worst person", "deserve only insults"
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
            "ruin everything", "nobody wants",
            "burden to this group", "stop talking"
        ],
    }

    for label, keywords in label_keywords.items():
        for word in keywords:
            if word in text:
                return label, 0.85, f"The comment contains {label} language."

    return "safe", 0.75, "No toxic content detected"


@app.post("/predict")
def predict(req: PredictRequest):
    label, confidence, explanation = classify_comment(req.comment)

    return {
        "label": label,
        "confidence": confidence,
        "explanation": explanation
    }


@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    results = []

    for comment in req.comments:
        label, confidence, explanation = classify_comment(comment)
        results.append({
            "comment": comment,
            "label": label,
            "confidence": confidence,
            "explanation": explanation
        })

    return {"results": results}


# ---------------------------
# Accurate Document Q&A
# ---------------------------
def clean_word(word: str):
    return word.lower().strip(".,?!:;()[]{}\"'")


def answer_from_document(document: str, question: str):
    question_words = [
        clean_word(word)
        for word in question.split()
        if len(clean_word(word)) > 3
    ]

    if not question_words:
        return "Not mentioned in the document."

    sentences = (
        document.replace("\n", " ")
        .replace("?", ".")
        .replace("!", ".")
        .split(".")
    )

    best_sentence = ""
    best_score = 0

    for sentence in sentences:
        sentence_clean = sentence.strip()
        sentence_lower = sentence_clean.lower()

        if not sentence_clean:
            continue

        score = sum(1 for word in question_words if word in sentence_lower)

        if score > best_score:
            best_score = score
            best_sentence = sentence_clean

    if best_score == 0:
        return "Not mentioned in the document."

    return best_sentence + "."


@app.post("/ask")
def ask(req: AskRequest):
    document = req.document.strip()
    question = req.question.strip()

    if not document:
        raise HTTPException(status_code=400, detail="Document cannot be empty")

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = answer_from_document(document, question)

    return {
        "answer": answer
    }