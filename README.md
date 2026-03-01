# Nativity_Project2026
# 🎙️ Nativity Accent Classification

A deep learning pipeline that classifies whether a speaker is a **native** or **non-native** speaker of a language, using audio recordings as input. The model combines **Wav2Vec2** embeddings with **MFCC** features and a language-aware embedding layer.

---

## 📌 Overview

This project tackles the binary classification problem of nativity detection from speech. Given an audio clip and the target language, the model predicts:

- `1` → Native Speaker
- `0` → Non-Native Speaker

---

## 🗂️ Project Structure

```
├── train.py          # Full training pipeline
├── dataset.csv       # CSV with audio URLs and labels
├── audio/            # Auto-downloaded audio files (created at runtime)
├── settings.json     # VSCode Python environment config (conda)
└── README.md
```

---

## ⚙️ Requirements

Install dependencies via conda or pip:

```bash
pip install torch torchaudio transformers librosa noisereduce scikit-learn pandas numpy tqdm requests gdown
```

Or with conda:

```bash
conda install pytorch torchaudio -c pytorch
pip install transformers librosa noisereduce scikit-learn tqdm gdown
```

> **Environment:** Python 3.8+, CPU compatible (CUDA optional)

---

## 📄 Dataset Format

The model expects a `dataset.csv` file with at least the following columns:

| Column | Description |
|---|---|
| `audio_url` | Direct URL or Google Drive link to `.wav` audio file |
| `language` | Language of the speaker (e.g., `english`, `french`) |
| `nativity_status` | Label: `native` / `non-native` (or `1` / `0`) |

> The `dp_id` column (if present) is automatically dropped to prevent data leakage.

---

## 🚀 How to Run

```bash
python train.py
```

The script will:
1. Load and shuffle `dataset.csv`
2. Download all audio files to the `audio/` directory
3. Convert and validate labels
4. Perform a language-wise fair 75/25 train/test split
5. Encode languages and preprocess audio
6. Train the model for 10 epochs
7. Evaluate and print final metrics

---

## 🏗️ Model Architecture

### Feature Extraction

| Feature | Description |
|---|---|
| **Wav2Vec2** (`facebook/wav2vec2-base`) | 768-dim contextual speech embeddings. CNN layers frozen; only last 2 transformer layers are fine-tuned |
| **MFCC** | 39-dim features (40 coefficients minus 0th), with Cepstral Mean Variance Normalization (CMVN) |
| **Language Embedding** | 8-dim learned embedding per language |

### Classifier

```
Input: [Wav2Vec2 (768) + MFCC (39) + Lang Embedding (8)] = 815-dim
→ Linear(815, 256) → ReLU → Dropout(0.3)
→ Linear(256, 1) → BCEWithLogitsLoss
```

### Audio Preprocessing

Each audio file is preprocessed with:
- Resampling to 16,000 Hz
- DC offset removal
- Peak normalization
- Spectral noise reduction (`noisereduce`)
- RMS normalization (target RMS = 0.1)

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Batch Size | 1 |
| Optimizer | AdamW |
| Classifier LR | `2e-4` |
| Wav2Vec2 fine-tune LR | `1e-5` |
| Loss Function | BCEWithLogitsLoss |
| Gradient Clipping | 1.0 |

---

## 🧪 Evaluation Metrics

After training, the model is evaluated on the held-out test set and reports:

- Accuracy
- Precision
- Recall
- F1 Score
- Full Classification Report
- Confusion Matrix

---

## 🔍 Inference

Use the `predict_audio()` function to classify a new audio file:

```python
predict_audio("path/to/audio.wav", "english")
```

Output:
```
Prediction: Native
Confidence: 0.87
```

---

## ⚖️ Fair Split Strategy

The train/test split is performed **per language** to ensure fairness across languages and classes:

- For each language, 25% of samples (per class) are allocated to the **training set**
- The remaining 75% form the **test set**
- If a class has fewer than 25% of the total language samples, all samples of that class go to training

---

## 🛠️ VSCode Setup

This project uses **conda** as the default Python environment and package manager (configured in `settings.json`). Open the project in VSCode and select your conda environment via the Python extension.

---

## 📝 Notes

- The model runs on **CPU by default**; CUDA is used automatically if available.
- Google Drive audio links are supported via `gdown`.
- Files that fail to download are automatically excluded from training/testing.

## 1 System Requirements
Minimum:

Windos 10/11 (64-bit)

8 GB RAM (16 GB recommended)

20 GB free disk space

Internet connection (first run downloads wav2vec model + audio)

Software you will install:

Anaconda

Python 3.10 environment

FFmpeg

VS Code (recommended) or Jupyter Notebook

## 2 Install Anaconda
Download:
https://www.anaconda.com/download

Choose:
Python 3.10 (64-bit)

During installation:
IMPORTANT:
✔ Check “Register Anaconda as system Python”
❌ Do NOT check “Add to PATH” (we activate manually)

Finish installation.

## 3 Create the Project Environment
Open Anaconda Prompt.

Create environment:

conda create -n nativity python=3.10 -y
Activate it:

conda activate nativity
You must now see:

(nativity) C:\Users\...
## 4 Install Required Libraries
Run all of these inside the nativity environment:

pip install torch torchaudio
pip install transformers
pip install librosa
pip install soundfile
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tqdm
pip install noisereduce
pip install audioread
pip install requests
This installs:

Deep learning (PyTorch + Wav2Vec2)

Audio processing

Evaluation metrics

Visualization

## 5 Install FFmpeg (VERY IMPORTANT)
Without this → audio loading and MFCC images will fail.

Step 1 — Download
Go to:

https://www.gyan.dev/ffmpeg/builds/

Download:
ffmpeg-release-essentials.zip

Step 2 — Extract
Extract to:

C:\ffmpeg
You must have:

C:\ffmpeg\bin\ffmpeg.exe
Step 3 — Add to PATH
Windows search → Environment Variables

Open: Edit the system environment variables

Click Environment Variables

Under System Variables → select Path

Click Edit → New

Add:

C:\ffmpeg\bin
Click OK → OK → OK.

Restart VS Code or terminal.

Step 4 — Verify
Open Anaconda Prompt:

ffmpeg -version
If it prints version → success.

## 6 Project Folder Structure
Create a folder:

Nativity_Project
Inside place:

train.py
dataset.csv
(Your CSV containing audio links)

## 7 Open Project in VS Code
Open VS Code

File → Open Folder → select Nativity_Project

Press Ctrl + Shift + P

Type:

Python: Select Interpreter
Choose:

Python (nativity)
This step is critical.

## 8 First Run (Important)
In VS Code terminal:

Activate environment again:

conda activate nativity
Run:

python train.py
## 9 What Happens During First Execution
The program will:

Read CSV

Download audio from links

Convert mp3 → wav automatically

Download wav2vec2 pretrained model (~360 MB) (only first time)

Split dataset (balanced per language)

Train 5 epochs

Select best epoch

Evaluate model

Generate outputs

First run may take 20–40 minutes.

Later runs are faster.

## 10 Output Files Generated
After completion you will get:

Nativity_Project/
│
├── audio/                        (downloaded audio files)
├── saved_model/
│     best_model.pth
│
├── mfcc_images/
│     sample_0.png
│     sample_1.png
│
├── evaluation_graphs/
│     confusion_matrix.png
│     roc_curve.png
│     precision_recall_curve.png
│     confidence_histogram.png
│     language_accuracy.png
│
├── nativity_predictions.csv
## 11 Meaning of Outputs
nativity_predictions.csv
Contains:

test sample

true label

predicted label

confidence score

mfcc_images
Visual acoustic representation of speech features.

evaluation_graphs
Shows:

fairness

classification ability

confidence calibration

per-language performance

best_model.pth
Your trained nativity detection model.

## 12 Running Again
After first run, audio and wav2vec are cached.

So next time just:

conda activate nativity
python train.py
No re-download needed.

## 13 Common Errors & Fixes
Error	Cause	Fix
audioread.NoBackendError	FFmpeg missing	install FFmpeg
ModuleNotFoundError seaborn	wrong environment	activate nativity
accuracy constant	frozen wav2vec	partial fine-tuning
audio not loading	broken link	remove that row
num_samples=0	dataset split failed	check column names
## 14 Final Notes
Your program now performs:

bias-controlled training

balanced language sampling

nativity classification

confidence estimation

fairness evaluation

acoustic visualization

This is a complete reproducible speech AI pipeline, not just a neural network script.

NOTE
--

Automated Predictions Generation: The predictions.csv file, containing the final predicted classifications and their corresponding confidence scores, is autonomously generated by the codebase upon completion of the inference pipeline.

Automated Visualizations: All visual data representations, including the MFCC spectrograms and statistical performance evaluation graphs, are systematically generated by the code during runtime to ensure a fully reproducible and automated workflow.


