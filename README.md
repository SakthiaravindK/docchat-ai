# docchat-ai
DocChat AI – Document Q&amp;A and Toxic Comment Detection using FastAPI, Streamlit and LoRA fine-tuning
# DocChat AI

## Overview
DocChat AI is a system for:
- Document Question Answering
- Toxic Comment Detection

## Features
- FastAPI backend
- Streamlit UI
- Toxic classification (rule-based)
- Document Q&A using sentence extraction
- Fine-tuning using LoRA (tiny-gpt2)
- Evaluation using F1 Score and ROUGE-L

## Project Structure
- api.py → backend
- ui.py → frontend
- train.py → fine-tuning
- evaluate.py → evaluation
- data/ → datasets

## How to Run

### Start API
python -m uvicorn api:app --reload


### Start UI

streamlit run ui.py


## Model Training

python train.py


## Evaluation

python evaluate.py


## Sample Output
- Q&A answers from document
- Toxic classification labels

## Author
Sakthi Aravind K

Save → then upload again.
