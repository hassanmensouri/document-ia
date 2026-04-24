from fastapi import FastAPI, UploadFile, File
import shutil
import os
import pickle
import requests

from utils import extract_text_from_file, smart_extract
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

with open("classifier.pkl", "rb") as f:
    clf = pickle.load(f)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_SUMMARY_MODEL = "csebuetnlp/mT5_multilingual_XLSum"


def summarize_with_hf(text):
    if not HF_TOKEN:
        return "HF_TOKEN is missing"

    url = f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    payload = {
        "inputs": "summarize: " + text[:5000],
        "parameters": {
            "max_length": 100,
            "min_length": 30,
            "do_sample": False
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        return f"Hugging Face error: {response.text}"

    data = response.json()

    if isinstance(data, list) and len(data) > 0:
        return data[0].get("summary_text", str(data[0]))

    return str(data)


@app.get("/")
def home():
    return {"message": "Document AI API is running"}


@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = extract_text_from_file(file_path)

        if not text.strip():
            return {"error": "No text extracted from document"}

        processed = smart_extract(text)

        emb = model.encode([processed])
        label = clf.predict(emb)[0]

        summary = summarize_with_hf(text)

        return {
            "classification": label,
            "summary": summary
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)