from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.get("/summarize/")
def summarize(text: str):
    return {"summary": summarizer(text, max_length=200, min_length=100, do_sample=False)[0]['summary_text']}
