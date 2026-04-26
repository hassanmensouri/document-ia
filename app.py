from fastapi import FastAPI, UploadFile, File
import shutil
import os
import pickle
import requests
import numpy as np
from huggingface_hub import InferenceClient

from utils import extract_text_from_file, smart_extract

app = FastAPI()

with open("classifier.pkl", "rb") as f:
    clf = pickle.load(f)

HF_TOKEN = os.getenv("HF_TOKEN")

HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
HF_SUMMARY_MODEL = "csebuetnlp/mT5_multilingual_XLSum"


def call_hf_api(url, payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise Exception(f"Hugging Face error: {response.text}")

    return response.json()


client = InferenceClient(
    provider="hf-inference",
    token=HF_TOKEN
)

def get_embedding(text):
    if not HF_TOKEN:
        raise Exception("HF_TOKEN is missing")

    data = client.feature_extraction(
        text[:2000],
        model=HF_EMBEDDING_MODEL
    )

    arr = np.array(data)

    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    elif arr.ndim == 3:
        arr = arr[0].mean(axis=0)

    return arr.reshape(1, -1)

def summarize_with_hf(text):
    if not HF_TOKEN:
        return "HF_TOKEN is missing"

    result = client.summarization(
        text[:5000],
        model=HF_SUMMARY_MODEL
    )

    return result.summary_text
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

        emb = get_embedding(processed)
        label = clf.predict(emb)[0]

        summary = summarize_with_hf(text)

        return {
            "classification": label,
            "summary": summary
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)