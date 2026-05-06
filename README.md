# Garbage Classification System (Flask + MobileNetV2)

A complete web-based garbage classification system using Transfer Learning with MobileNetV2.

## Features

- Classifies waste into 5 categories:
  - Plastic
  - Paper
  - Glass
  - Metal
  - Organic Waste
- Transfer learning with TensorFlow/Keras MobileNetV2
- Input image size: 224x224
- Data preprocessing + augmentation
- 70/15/15 train/validation/test split
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- Saves training curves:
  - Training vs validation accuracy
  - Training vs validation loss
- Responsive Flask web app (mobile + desktop)
- Image preview, loading animation, prediction confidence

## Project Structure

```text
Image Processing System/
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
├── .gitignore
├── model/
│   └── train.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── main.js
├── artifacts/                      # Created after training
│   ├── garbage_classifier.keras
│   ├── best_model.keras
│   ├── class_names.json
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   └── loss_curve.png
└── Garbage classification/
    └── Garbage classification/
        ├── cardboard/
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        └── trash/
```

## Dataset Note

Your raw dataset has folders:
`cardboard, glass, metal, paper, plastic, trash`

This project maps them to the required 5 classes:

- `plastic` -> `plastic`
- `paper` + `cardboard` -> `paper`
- `glass` -> `glass`
- `metal` -> `metal`
- `trash` -> `organic_waste`

## 1) Local Setup

### Windows PowerShell

```powershell
cd "c:\Users\Mythra\Desktop\Image Processing System"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Train the Model

```powershell
python model/train.py --dataset-dir "Garbage classification/Garbage classification" --output-dir artifacts --prepared-dir artifacts/splits --epochs 25 --batch-size 32 --learning-rate 0.001
```

Training uses:

- MobileNetV2 transfer learning
- Input size: 224x224
- Augmentation: flip, rotation, zoom
- Early stopping (`patience=5`)
- Default epochs: 25 (within requested 20-30)

## 3) Run the Web App

```powershell
python app.py
```

Open:

- `http://127.0.0.1:5000`

## 4) Evaluation Outputs

After training, check `artifacts/`:

- `metrics.json`: accuracy, precision, recall, f1, full classification report
- `confusion_matrix.png`
- `accuracy_curve.png`
- `loss_curve.png`

## 5) Deploy

### Render

1. Push this project to GitHub.
2. Create a new **Web Service** on Render.
3. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Deploy.

### Railway

1. Create a new Railway project from your GitHub repo.
2. Railway auto-detects `Procfile`.
3. Ensure env uses Python 3.11 (from `runtime.txt`).
4. Deploy.

### Hugging Face Spaces (Docker or Gradio/Static options)

For Flask deployment, easiest route is **Docker Space**:

1. Create a Docker Space.
2. Add your app files and a Dockerfile.
3. Start with gunicorn command to expose `$PORT`.

## Beginner Tips

- If the app says model is missing, run training first.
- Keep the same `artifacts/` files with your app during deployment.
- For best results, upload clear images with one main object.
