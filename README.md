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

A binary image classifier that predicts whether an image is a real photograph or an AI-generated (latent-diffusion) image, with a CNN trained on the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images). Method based on **Bird & Lotfi, "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images," IEEE Access, 2024.**

**Live demo:** `https://<your-hf-username>-cifake-detector.hf.space` &nbsp;·&nbsp; **Repo:** `https://github.com/UmangSaluja10/cifake-detector`

---

## What it does

Upload an image → the model resizes it to 32×32, runs it through a CNN, and returns a `REAL` / `FAKE` label with a confidence score.

| | |
|---|---|
| **Model** | 2× Conv2D(32) + MaxPool → Dense(64) → sigmoid |
| **Dataset** | CIFAKE — 120,000 images (60k real CIFAR-10 + 60k Stable Diffusion 1.4 fakes) |
| **Train / test split** | 100,000 / 20,000 images |
| **Validation accuracy** | ~93.8% |
| **Validation precision / recall** | ~94.3% / ~96.9% |

## Project structure

```
cifake-project/
├── model/
│   └── train_model.py      # trains the CNN on the CIFAKE dataset (run in Colab)
├── app/
│   ├── main.py              # FastAPI backend — /predict, /health, serves the UI
│   ├── static/
│   │   └── index.html       # frontend (vanilla HTML/CSS/JS)
│   └── saved_model/
│       └── cifake_cnn.h5    # trained model weights
├── requirements.txt
├── Dockerfile
└── README.md
```

## Tech stack

- **Model:** TensorFlow / Keras, trained on Google Colab (free T4 GPU)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** single-page HTML, CSS, vanilla JS (no build step, no framework)
- **Deployment:** Docker container on Hugging Face Spaces (free tier)

## Run it locally

```bash
git clone https://github.com/UmangSaluja10/cifake-detector.git
cd cifake-detector

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000**.

## Retrain the model

The CNN is trained on the 120,000-image CIFAKE dataset, which is too large to
fetch from this repo's own scripts at request time — train it in Colab
instead. Full steps are in `model/train_model.py`:

1. Open [Colab](https://colab.research.google.com) → **Runtime → Change runtime type → T4 GPU**.
2. Upload a Kaggle API token (`kaggle.json`) and run the setup cell.
3. Paste in and run `model/train_model.py`.
4. Download the resulting `cifake_cnn.h5` and place it at `app/saved_model/cifake_cnn.h5`.

## API reference

### `POST /predict`

Multipart form upload.

| Field | Type | Required |
|---|---|---|
| `file` | image (PNG/JPG) | yes |

**Response**

```json
{ "label": "REAL", "confidence": 96.42 }
```

### `GET /health`

```json
{ "status": "ok" }
```

## Deployment

This repo builds as a single Docker service (backend + frontend together) and is deployed on [Hugging Face Spaces](https://huggingface.co/spaces):

```bash
git remote add hf https://huggingface.co/spaces/<your-hf-username>/cifake-detector
git push hf main
```

The Space rebuilds automatically on every push. `app_port: 7860` in this README's frontmatter tells Spaces which port the container listens on.

## Known limitations

- **Narrow training domain.** The model only ever saw CIFAR-10's 10 object categories (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck) at 32×32 resolution. Images outside those categories, at higher resolution, or produced by newer generators (Midjourney, Imagen, DALL·E 3, etc.) are out-of-distribution and may be misclassified — this is disclosed directly on the site.
- **Single-generator training.** Fakes were generated only with Stable Diffusion 1.4; generalization to other generators is untested.
- **No robustness training.** The model wasn't trained against compression, resizing, or cropping, so accuracy likely drops on images that have passed through social platforms (consistent with findings in the 2026 NTIRE robustness challenge on AI-image detection).

## Future enhancements

- Train on a broader, multi-category, multi-generator dataset for real-world generalization.
- Add Grad-CAM visual explanations to the API response (see the original CIFAKE paper's explainability approach).
- Add robustness training against JPEG compression, resizing, and cropping.

## Credit

J. J. Bird and A. Lotfi, "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images," *IEEE Access*, vol. 12, pp. 15642–15650, 2024.