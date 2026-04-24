from fastapi import FastAPI, UploadFile, File
import shutil
import os
import pickle

from utils import extract_text_from_file, smart_extract, summarize_text
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = FastAPI()

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

with open("classifier.pkl", "rb") as f:
    clf = pickle.load(f)

summarizer = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum"
)

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

        summary = summarize_text(text, summarizer)

        return {
            "classification": label,
            "summary": summary
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)