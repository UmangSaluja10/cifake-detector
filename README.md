---
title: CIFAKE Detector
emoji: 🕵️
colorFrom: gray
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# CIFAKE Detector

A binary image classifier that predicts whether an image is a real photograph
or an AI-generated (latent-diffusion) image, based on the CIFAKE approach by
Bird & Lotfi (2024, IEEE Access). Upload an image in the browser and get a
REAL / FAKE label with a confidence score.

## Project structure

```
cifake-project/
├── model/
│   └── train_model.py      # trains the CNN on the CIFAKE dataset (run in Colab)
├── app/
│   ├── main.py              # FastAPI backend: /predict, /health, serves the UI
│   ├── static/
│   │   └── index.html       # frontend (vanilla HTML/JS)
│   └── saved_model/
│       └── cifake_cnn.h5    # trained model weights (add this yourself, see below)
├── requirements.txt
├── Dockerfile
└── README.md
```

## 1. Train the model

The model is trained in Google Colab (free T4 GPU) because it needs the
120,000-image CIFAKE dataset from Kaggle. Open `model/train_model.py` for
full step-by-step instructions. When it finishes, download `cifake_cnn.h5`
and place it at `app/saved_model/cifake_cnn.h5` in this repo.

## 2. Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 in your browser.

## 3. Deploy

This repo is set up to deploy as a single Docker service on
[Hugging Face Spaces](https://huggingface.co/spaces) (free tier, no card
required). See the deployment steps in the chat / project notes for the
exact commands to push this repo there and to GitHub.

## API

`POST /predict` — multipart form with a `file` field containing an image.
Returns:

```json
{ "label": "REAL", "confidence": 96.42 }
```

## Credit

Method based on: J. J. Bird and A. Lotfi, "CIFAKE: Image Classification and
Explainable Identification of AI-Generated Synthetic Images," IEEE Access,
vol. 12, pp. 15642–15650, 2024.
