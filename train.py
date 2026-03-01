# ==========================================================
# NATIVITY ACCENT CLASSIFICATION - FULL IMPLEMENTATION
# CPU Compatible | VSCode | Anaconda | Jupyter
# Includes automatic audio download from CSV links
# ==========================================================

import os
import requests
import gdown
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import noisereduce as nr
import math

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
os.makedirs("saved_model", exist_ok=True)
# ----------------------------------------------------------
# DEVICE
# ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ----------------------------------------------------------
# LOAD CSV
# ----------------------------------------------------------
df = pd.read_csv("dataset.csv")

# remove id leakage
if "dp_id" in df.columns:
    df = df.drop(columns=["dp_id"])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------------------------------------
# DOWNLOAD AUDIO FROM LINKS
# ----------------------------------------------------------
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def download_audio(url, index):
    try:
        filename = f"{AUDIO_DIR}/sample_{index}.wav"

        if os.path.exists(filename):
            return filename

        # Google Drive
        if "drive.google.com" in url:
            if "id=" in url:
                file_id = url.split("id=")[1]
            elif "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
            else:
                raise Exception("Unsupported Drive link")

            gdown.download(f"https://drive.google.com/uc?id={file_id}",
                           filename, quiet=False)

        # Normal HTTP
        else:
            response = requests.get(url, stream=True, timeout=60)
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    if chunk:
                        f.write(chunk)

        return filename

    except Exception as e:
        print("Download failed:", url)
        return None

print("\nDownloading audio dataset...\n")
local_paths = []
valid_rows = []

for i, row in df.iterrows():
    path = download_audio(row["audio_url"], i)
    if path is not None:
        local_paths.append(path)
        valid_rows.append(i)

df = df.iloc[valid_rows].reset_index(drop=True)
df["audio_url"] = local_paths

print("Audio download complete.\n")

# REMOVE FILES THAT FAILED TO DOWNLOAD
# --------------------------------------------------

valid_rows = []

for i, row in df.iterrows():
    if os.path.exists(row["audio_url"]):
        valid_rows.append(i)

df = df.loc[valid_rows].reset_index(drop=True)

print("Usable audio files:", len(df))

# --------------------------------------------------
# CONVERT TEXT LABELS TO BINARY (CRITICAL FIX)
# --------------------------------------------------

# --------------------------------------------------
# SAFE LABEL CONVERSION (FINAL FIX)
# --------------------------------------------------

df["nativity_status"] = df["nativity_status"].astype(str).str.strip().str.lower()

print("Original label values:", df["nativity_status"].unique())

def convert_label(x):

    if x in ["native", "native speaker", "n", "yes", "1"]:
        return 1

    elif x in ["non-native", "non native", "non_native", "nn", "no", "0"]:
        return 0

    else:
        return None

df["nativity_status"] = df["nativity_status"].apply(convert_label)

# remove unknown labels
df = df.dropna(subset=["nativity_status"])
df["nativity_status"] = df["nativity_status"].astype(int)

print("Converted labels:", df["nativity_status"].unique())
# ----------------------------------------------------------
# BALANCED Training and train -test - split(25% RULE)
# ----------------------------------------------------------
# ==========================================================
# NEW FAIRNESS-BASED TRAIN / TEST SPLIT
# ==========================================================
# ==========================================================
# CORRECT FAIRNESS-BASED TRAIN / TEST SPLIT
# ==========================================================

# ==========================================================
# FINAL LANGUAGE-WISE FAIR SPLIT
# ==========================================================

# ==========================================================
# FINAL LANGUAGE-COUNT BASED 25% SPLIT
# ==========================================================



# ==========================================================
# FINAL LANGUAGE-COUNT BASED 25% SPLIT
# ==========================================================

import math
import pandas as pd

train_parts = []
test_parts = []

for lang in sorted(df["language"].unique()):

    # ----- Separate language -----
    lang_df = df[df["language"] == lang]

    T = len(lang_df)  # total samples in this language
    k = math.ceil(0.25 * T)  # 25% of language size

    # ----- Separate classes -----
    native_df = lang_df[lang_df["nativity_status"] == 1]
    nonnative_df = lang_df[lang_df["nativity_status"] == 0]

    # ================= NATIVE =================
    if len(native_df) < 0.25 * T:
        native_train = native_df
        native_test = pd.DataFrame(columns=df.columns)
    else:
        native_train = native_df.sample(n=min(k, len(native_df)), random_state=42)
        native_test = native_df.drop(native_train.index)

    # ================= NON-NATIVE =================
    if len(nonnative_df) < 0.25 * T:
        nonnative_train = nonnative_df
        nonnative_test = pd.DataFrame(columns=df.columns)
    else:
        nonnative_train = nonnative_df.sample(n=min(k, len(nonnative_df)), random_state=42)
        nonnative_test = nonnative_df.drop(nonnative_train.index)

    # Combine per-language
    train_lang = pd.concat([native_train, nonnative_train])
    test_lang = pd.concat([native_test, nonnative_test])

    train_parts.append(train_lang)
    test_parts.append(test_lang)

# Combine all languages
train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

# Shuffle
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("Languages in training:", len(train_df["language"].unique()))
print("Languages in testing:", len(test_df["language"].unique()))
# train_indices = []
# test_indices = []

# for lang in df["language"].unique():

#     subset = df[df["language"] == lang]
#     indices = subset.index.tolist()

#     native_idx = subset[subset["nativity_status"] == 1].index.tolist()
#     non_idx = subset[subset["nativity_status"] == 0].index.tolist()

#     # identify minority and majority
#     if len(native_idx) < len(non_idx):
#         minority = native_idx
#         majority = non_idx
#     else:
#         minority = non_idx
#         majority = native_idx

#     total_lang = len(subset)

#     # -------- CASE 1: Minority < 25% of total --------
#     if len(minority) < 0.25 * total_lang:

#         # take ALL minority
#         selected_minority = minority

#         # take equal number of majority
#         selected_majority = majority[:len(selected_minority)]

#     # -------- CASE 2: Minority >= 25% --------
#     else:

#         sample_size = int(0.25 * total_lang)

#         selected_minority = minority[:sample_size]
#         selected_majority = majority[:sample_size]

#     # add to training
#     train_lang = selected_minority + selected_majority
#     train_indices.extend(train_lang)

    # rest goes to testing
#     remaining = list(set(indices) - set(train_lang))
#     test_indices.extend(remaining)

# # Create datasets
# train_df = df.loc[train_indices].reset_index(drop=True)
# test_df = df.loc[test_indices].reset_index(drop=True)

# print("Training samples:", len(train_df))
# print("Testing samples:", len(test_df))
# def balanced_sampling(df):
#     balanced = []

#     for lang in df["language"].unique():
#         subset = df[df["language"] == lang]

#         native = subset[subset["native_label"] == 1]
#         non_native = subset[subset["native_label"] == 0]

#         if len(native) < len(non_native):
#             minority, majority = native, non_native
#         else:
#             minority, majority = non_native, native

#         if len(minority) < 0.25 * len(majority):
#             majority_sample = majority.sample(int(0.25 * len(majority)), random_state=42)
#             sampled = pd.concat([minority, majority_sample])
#         else:
#             minority_sample = minority.sample(int(0.25 * len(minority)), random_state=42)
#             majority_sample = majority.sample(int(0.25 * len(majority)), random_state=42)
#             sampled = pd.concat([minority_sample, majority_sample])

#         balanced.append(sampled)

#     return pd.concat(balanced)

# df = balanced_sampling(df)

# ----------------------------------------------------------
# TRAIN TEST INDIVIDUAL SHUFFLING
# ----------------------------------------------------------
# train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
# train_df, test_df = train_test_split(df, test_size=0.5,
#                                      stratify=df["native_label"],
#                                      random_state=42)

# ----------------------------------------------------------
# LANGUAGE ENCODING
# ----------------------------------------------------------
lang_encoder = LabelEncoder()
train_df["lang_id"] = lang_encoder.fit_transform(train_df["language"])
test_df["lang_id"] = lang_encoder.transform(test_df["language"])



# ----------------------------------------------------------
# AUDIO PREPROCESSING
# ----------------------------------------------------------
def preprocess_audio(path):

    audio, sr = librosa.load(path, sr=16000)

    # remove DC offset
    audio = audio - np.mean(audio)

    # peak normalization (safe, fast)
    max_val = np.max(np.abs(audio)) + 1e-9
    audio = audio / max_val

    # optional mild noise reduction (lightweight spectral gating)
    audio = nr.reduce_noise(y=audio, sr=16000, prop_decrease=0.8)

    # RMS normalization (critical for wav2vec stability)
    rms = np.sqrt(np.mean(audio**2) + 1e-9)
    target_rms = 0.1
    audio = audio * (target_rms / rms)

    return audio

# ----------------------------------------------------------
# MFCC
# ----------------------------------------------------------
def extract_mfcc(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=16000,
        n_mfcc=40,
        n_fft=400,
        hop_length=160
    )

    # Remove 0th coefficient (energy)
    mfcc = mfcc[1:]

    # Cepstral Mean Variance Normalization (fast & essential)
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True) + 1e-6
    mfcc = (mfcc - mean) / std

    # Temporal pooling
    mfcc = np.mean(mfcc, axis=1)

    return mfcc.astype(np.float32)

# ----------------------------------------------------------
# WAV2VEC
# ----------------------------------------------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

# Freeze CNN feature extractor
for param in wav2vec.feature_extractor.parameters():
    param.requires_grad = False

# Freeze first 10 transformer layers
for layer in wav2vec.encoder.layers[:10]:
    for param in layer.parameters():
        param.requires_grad = False

# Only last 2 transformer layers trainable
for layer in wav2vec.encoder.layers[10:]:
    for param in layer.parameters():
        param.requires_grad = True
# ----------------------------------------------------------
# DATASET CLASS
# ----------------------------------------------------------
def mean_pool_wav2vec(hidden):

    # average across time
    pooled = hidden.mean(dim=0)

    # L2 normalization (VERY IMPORTANT)
    pooled = pooled / (pooled.norm(p=2) + 1e-9)

    return pooled
class NativityDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio = preprocess_audio(row["audio_url"])

        inputs = processor(audio, sampling_rate=16000,
                           return_tensors="pt", padding=True)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        
        outputs = wav2vec(**inputs)

        hidden = outputs.last_hidden_state.squeeze(0)

# ---- POOLING (CRITICAL FIX) ----
        wav_emb = mean_pool_wav2vec(hidden)
        mfcc = torch.tensor(extract_mfcc(audio), dtype=torch.float32)

# force it to be a flat vector
        mfcc = mfcc.view(-1)

        mfcc = mfcc.to(device)

        lang_id = torch.tensor(row["lang_id"], dtype=torch.long).to(device)

        label = torch.tensor(row["nativity_status"], dtype=torch.float32).to(device)

        return wav_emb, mfcc, lang_id, label

# ----------------------------------------------------------
# ATTENTION POOLING
# ----------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=0)
        return torch.sum(weights * x, dim=0)



# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------
class NativityModel(nn.Module):
    def __init__(self, lang_count):
        super().__init__()

       
        self.lang_embed = nn.Embedding(lang_count, 8)

        wav_dim = 768
        mfcc_dim = 39
        lang_dim = 8

        input_dim = wav_dim + mfcc_dim + lang_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, wav_emb, mfcc, lang_id):
        wav_vec = wav_emb
        lang_vec = self.lang_embed(lang_id)

        combined = torch.cat([wav_vec, mfcc, lang_vec], dim=-1)
        x = self.dropout(self.relu(self.fc1(combined)))
        return self.fc2(x)


# ----------------------------------------------------------
# TRAINING
# ----------------------------------------------------------
train_loader = DataLoader(NativityDataset(train_df), batch_size=1, shuffle=True)
test_loader = DataLoader(NativityDataset(test_df), batch_size=1,shuffle=False)

model = NativityModel(len(lang_encoder.classes_)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW([
    {"params": model.parameters(), "lr": 2e-4},
    {"params": filter(lambda p: p.requires_grad, wav2vec.parameters()), "lr": 1e-5}
])

print("\nTraining started...\n")
best_accuracy = 0
best_epoch = 0
for epoch in range(5):

    model.train()
    total_loss = 0

    for wav_emb, mfcc, lang_id, label in tqdm(train_loader):

        # ensure device consistency
        wav_emb = wav_emb.to(device)
        mfcc = mfcc.to(device)
        lang_id = lang_id.to(device)
        label = label.to(device)

        # forward
        logits = model(wav_emb, mfcc, lang_id)

        # compute loss
        loss = criterion(logits.view(-1), label.view(-1))

        # backward
        loss.backward()

        # prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update weights
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")
    # ================= EVALUATION =================
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for wav_emb, mfcc, lang_id, label in test_loader:

            wav_emb = wav_emb.to(device)
            mfcc = mfcc.to(device)
            lang_id = lang_id.to(device)
            label = label.to(device)

            logits = model(wav_emb, mfcc, lang_id)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds.view(-1) == label.view(-1)).sum().item()
            total += label.size(0)

    accuracy = correct / total
  

# ----------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------
print("\nEvaluating model...\n")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for wav_emb, mfcc, lang_id, label in test_loader:
        logits = model(wav_emb, mfcc, lang_id)
        probs = torch.softmax(logits, dim=-1)

        pred = torch.argmax(probs).item()

        if pred == label.item():
            correct += 1
        total += 1

print("FINAL ACCURACY:", correct/total)
# print("Mean probability:", probs.mean().item())
print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")
# ----------------------------------------------------------
# PREDICTION FUNCTION
# ----------------------------------------------------------
print(f"Epoch {epoch+1} Test Accuracy: {accuracy:.4f}")
# ---- SAVE BEST MODEL ----
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_epoch = epoch + 1
    torch.save(model.state_dict(), "saved_model/best_model.pth")
    print("Best model updated and saved.")
def predict_audio(path, language):
    audio = preprocess_audio(path)


    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = wav2vec(**inputs)

    wav_emb = outputs.last_hidden_state.squeeze(0)
    mfcc = torch.tensor(extract_mfcc(audio), dtype=torch.float32).to(device)

    lang_id = torch.tensor(lang_encoder.transform([language])[0]).to(device)

    logits = model(wav_emb, mfcc, lang_id)
    probs = torch.softmax(logits, dim=-1)

    print("\nPrediction:", "Native" if torch.argmax(probs)==1 else "Non-Native")
    print("Confidence:", torch.max(probs).item())


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,roc_curve,auc,precision_recall_curve
os.makedirs("mfcc_images", exist_ok=True)
os.makedirs("evaluation_graphs", exist_ok=True)
def save_mfcc_image(audio_path, file_id):
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40,
            n_fft=1024,
            hop_length=512
        )

        # Remove 0th coefficient (loudness bias removal)
        mfcc = mfcc[1:, :]

        # Normalize same as training
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        plt.figure(figsize=(10,4))
        librosa.display.specshow(mfcc, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title("MFCC")
        plt.tight_layout()

        plt.savefig(f"mfcc_images/sample_{file_id}.png")
        plt.close()

    except Exception as e:
        print("MFCC image error:", e)
print(f"\nLoading best model from epoch {best_epoch} with accuracy {best_accuracy:.4f}")
model.load_state_dict(torch.load("saved_model/best_model.pth", map_location=device))
model.eval()
print("\n========== FINAL TEST EVALUATION ==========\n")

all_labels = []
all_preds = []
all_probs = []
all_languages = []
all_paths = []

with torch.no_grad():
    for idx, (wav_emb, mfcc, lang_id, label) in enumerate(test_loader):

        wav_emb = wav_emb.to(device)
        mfcc = mfcc.to(device)
        lang_id = lang_id.to(device)
        label = label.to(device)

        logits = model(wav_emb, mfcc, lang_id)

        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

        all_labels.append(label.item())
        all_preds.append(pred)
        all_probs.append(prob)

        row = test_df.iloc[idx]
        all_languages.append(row["language"])
        all_paths.append(row["audio_url"])

        # Save MFCC images (limit to first 15 samples)
        if idx < 15:
            save_mfcc_image(row["audio_url"], idx)
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall    = recall_score(all_labels, all_preds)
f1        = f1_score(all_labels, all_preds)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=["Non-Native","Native"]))

print("\nConfusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))
results_df = pd.DataFrame({
    "audio_path": all_paths,
    "language": all_languages,
    "true_label": all_labels,
    "predicted_label": all_preds,
    "confidence_score": all_probs
})

results_df["true_label"] = results_df["true_label"].map({1:"native",0:"non-native"})
results_df["predicted_label"] = results_df["predicted_label"].map({1:"native",0:"non-native"})

results_df.to_csv("nativity_predictions.csv", index=False)

print("\nPredictions saved to nativity_predictions.csv")
results_df = pd.DataFrame({
    "audio_path": all_paths,
    "language": all_languages,
    "true_label": all_labels,
    "predicted_label": all_preds,
    "confidence_score": all_probs
})

results_df["true_label"] = results_df["true_label"].map({1:"native",0:"non-native"})
results_df["predicted_label"] = results_df["predicted_label"].map({1:"native",0:"non-native"})

results_df.to_csv("nativity_predictions.csv", index=False)

print("\nPredictions saved to nativity_predictions.csv")
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Native','Native'],
            yticklabels=['Non-Native','Native'])

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("evaluation_graphs/confusion_matrix.png")
plt.close()
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation_graphs/roc_curve.png")
plt.close()
precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("evaluation_graphs/precision_recall_curve.png")
plt.close()
plt.figure(figsize=(6,4))
plt.hist(all_probs, bins=20)
plt.xlabel("Confidence Score (Probability of Native)")
plt.ylabel("Samples")
plt.title("Confidence Score Distribution")
plt.tight_layout()
plt.savefig("evaluation_graphs/confidence_histogram.png")
plt.close()
language_results = {}

for i in range(len(all_labels)):
    lang = all_languages[i]

    if lang not in language_results:
        language_results[lang] = {"correct":0, "total":0}

    if all_labels[i] == all_preds[i]:
        language_results[lang]["correct"] += 1

    language_results[lang]["total"] += 1

langs = []
accs = []

for lang in language_results:
    acc = language_results[lang]["correct"] / language_results[lang]["total"]
    langs.append(lang)
    accs.append(acc)

plt.figure(figsize=(10,5))
plt.bar(langs, accs)
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Accuracy per Language")
plt.tight_layout()
plt.savefig("evaluation_graphs/language_accuracy.png")
plt.close()
train.py
Displaying train.py.
import requests
import gdown
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import noisereduce as nr
import math

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ----------------------------------------------------------
# DEVICE
# ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ----------------------------------------------------------
# LOAD CSV
# ----------------------------------------------------------
df = pd.read_csv("dataset.csv")

# remove id leakage
if "dp_id" in df.columns:
    df = df.drop(columns=["dp_id"])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------------------------------------
# DOWNLOAD AUDIO FROM LINKS
# ----------------------------------------------------------
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def download_audio(url, index):
    try:
        filename = f"{AUDIO_DIR}/sample_{index}.wav"

        if os.path.exists(filename):
            return filename

        # Google Drive
        if "drive.google.com" in url:
            if "id=" in url:
                file_id = url.split("id=")[1]
            elif "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
            else:
                raise Exception("Unsupported Drive link")

            gdown.download(f"https://drive.google.com/uc?id={file_id}",
                           filename, quiet=False)

        # Normal HTTP
        else:
            response = requests.get(url, stream=True, timeout=60)
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    if chunk:
                        f.write(chunk)

        return filename

    except Exception as e:
        print("Download failed:", url)
        return None

print("\nDownloading audio dataset...\n")
local_paths = []
valid_rows = []

for i, row in df.iterrows():
    path = download_audio(row["audio_url"], i)
    if path is not None:
        local_paths.append(path)
        valid_rows.append(i)

df = df.iloc[valid_rows].reset_index(drop=True)
df["audio_url"] = local_paths

print("Audio download complete.\n")

# REMOVE FILES THAT FAILED TO DOWNLOAD
# --------------------------------------------------

valid_rows = []

for i, row in df.iterrows():
    if os.path.exists(row["audio_url"]):
        valid_rows.append(i)

df = df.loc[valid_rows].reset_index(drop=True)

print("Usable audio files:", len(df))

# --------------------------------------------------
# CONVERT TEXT LABELS TO BINARY (CRITICAL FIX)
# --------------------------------------------------

# --------------------------------------------------
# SAFE LABEL CONVERSION (FINAL FIX)
# --------------------------------------------------

df["nativity_status"] = df["nativity_status"].astype(str).str.strip().str.lower()

print("Original label values:", df["nativity_status"].unique())

def convert_label(x):

    if x in ["native", "native speaker", "n", "yes", "1"]:
        return 1

    elif x in ["non-native", "non native", "non_native", "nn", "no", "0"]:
        return 0

    else:
        return None

df["nativity_status"] = df["nativity_status"].apply(convert_label)

# remove unknown labels
df = df.dropna(subset=["nativity_status"])
df["nativity_status"] = df["nativity_status"].astype(int)

print("Converted labels:", df["nativity_status"].unique())
# ----------------------------------------------------------
# BALANCED Training and train -test - split(25% RULE)
# ----------------------------------------------------------
# ==========================================================
# NEW FAIRNESS-BASED TRAIN / TEST SPLIT
# ==========================================================
# ==========================================================
# CORRECT FAIRNESS-BASED TRAIN / TEST SPLIT
# ==========================================================

# ==========================================================
# FINAL LANGUAGE-WISE FAIR SPLIT
# ==========================================================

# ==========================================================
# FINAL LANGUAGE-COUNT BASED 25% SPLIT
# ==========================================================



# ==========================================================
# FINAL LANGUAGE-COUNT BASED 25% SPLIT
# ==========================================================

import math
import pandas as pd

train_parts = []
test_parts = []

for lang in sorted(df["language"].unique()):

    # ----- Separate language -----
    lang_df = df[df["language"] == lang]

    T = len(lang_df)  # total samples in this language
    k = math.ceil(0.25 * T)  # 25% of language size

    # ----- Separate classes -----
    native_df = lang_df[lang_df["nativity_status"] == 1]
    nonnative_df = lang_df[lang_df["nativity_status"] == 0]

    # ================= NATIVE =================
    if len(native_df) < 0.25 * T:
        native_train = native_df
        native_test = pd.DataFrame(columns=df.columns)
    else:
        native_train = native_df.sample(n=min(k, len(native_df)), random_state=42)
        native_test = native_df.drop(native_train.index)

    # ================= NON-NATIVE =================
    if len(nonnative_df) < 0.25 * T:
        nonnative_train = nonnative_df
        nonnative_test = pd.DataFrame(columns=df.columns)
    else:
        nonnative_train = nonnative_df.sample(n=min(k, len(nonnative_df)), random_state=42)
        nonnative_test = nonnative_df.drop(nonnative_train.index)

    # Combine per-language
    train_lang = pd.concat([native_train, nonnative_train])
    test_lang = pd.concat([native_test, nonnative_test])

    train_parts.append(train_lang)
    test_parts.append(test_lang)

# Combine all languages
train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

# Shuffle
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("Languages in training:", len(train_df["language"].unique()))
print("Languages in testing:", len(test_df["language"].unique()))
# train_indices = []
# test_indices = []

# for lang in df["language"].unique():

#     subset = df[df["language"] == lang]
#     indices = subset.index.tolist()

#     native_idx = subset[subset["nativity_status"] == 1].index.tolist()
#     non_idx = subset[subset["nativity_status"] == 0].index.tolist()

#     # identify minority and majority
#     if len(native_idx) < len(non_idx):
#         minority = native_idx
#         majority = non_idx
#     else:
#         minority = non_idx
#         majority = native_idx

#     total_lang = len(subset)

#     # -------- CASE 1: Minority < 25% of total --------
#     if len(minority) < 0.25 * total_lang:

#         # take ALL minority
#         selected_minority = minority

#         # take equal number of majority
#         selected_majority = majority[:len(selected_minority)]

#     # -------- CASE 2: Minority >= 25% --------
#     else:

#         sample_size = int(0.25 * total_lang)

#         selected_minority = minority[:sample_size]
#         selected_majority = majority[:sample_size]

#     # add to training
#     train_lang = selected_minority + selected_majority
#     train_indices.extend(train_lang)

    # rest goes to testing
#     remaining = list(set(indices) - set(train_lang))
#     test_indices.extend(remaining)

# # Create datasets
# train_df = df.loc[train_indices].reset_index(drop=True)
# test_df = df.loc[test_indices].reset_index(drop=True)

# print("Training samples:", len(train_df))
# print("Testing samples:", len(test_df))
# def balanced_sampling(df):
#     balanced = []

#     for lang in df["language"].unique():
#         subset = df[df["language"] == lang]

#         native = subset[subset["native_label"] == 1]
#         non_native = subset[subset["native_label"] == 0]

#         if len(native) < len(non_native):
#             minority, majority = native, non_native
#         else:
#             minority, majority = non_native, native

#         if len(minority) < 0.25 * len(majority):
#             majority_sample = majority.sample(int(0.25 * len(majority)), random_state=42)
#             sampled = pd.concat([minority, majority_sample])
#         else:
#             minority_sample = minority.sample(int(0.25 * len(minority)), random_state=42)
#             majority_sample = majority.sample(int(0.25 * len(majority)), random_state=42)
#             sampled = pd.concat([minority_sample, majority_sample])

#         balanced.append(sampled)

#     return pd.concat(balanced)

# df = balanced_sampling(df)

# ----------------------------------------------------------
# TRAIN TEST INDIVIDUAL SHUFFLING
# ----------------------------------------------------------
# train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
# train_df, test_df = train_test_split(df, test_size=0.5,
#                                      stratify=df["native_label"],
#                                      random_state=42)

# ----------------------------------------------------------
# LANGUAGE ENCODING
# ----------------------------------------------------------
lang_encoder = LabelEncoder()
train_df["lang_id"] = lang_encoder.fit_transform(train_df["language"])
test_df["lang_id"] = lang_encoder.transform(test_df["language"])



# ----------------------------------------------------------
# AUDIO PREPROCESSING
# ----------------------------------------------------------
def preprocess_audio(path):

    audio, sr = librosa.load(path, sr=16000)

    # remove DC offset
    audio = audio - np.mean(audio)

    # peak normalization (safe, fast)
    max_val = np.max(np.abs(audio)) + 1e-9
    audio = audio / max_val

    # optional mild noise reduction (lightweight spectral gating)
    audio = nr.reduce_noise(y=audio, sr=16000, prop_decrease=0.8)

    # RMS normalization (critical for wav2vec stability)
    rms = np.sqrt(np.mean(audio**2) + 1e-9)
    target_rms = 0.1
    audio = audio * (target_rms / rms)

    return audio

# ----------------------------------------------------------
# MFCC
# ----------------------------------------------------------
def extract_mfcc(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=16000,
        n_mfcc=40,
        n_fft=400,
        hop_length=160
    )

    # Remove 0th coefficient (energy)
    mfcc = mfcc[1:]

    # Cepstral Mean Variance Normalization (fast & essential)
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True) + 1e-6
    mfcc = (mfcc - mean) / std

    # Temporal pooling
    mfcc = np.mean(mfcc, axis=1)

    return mfcc.astype(np.float32)

# ----------------------------------------------------------
# WAV2VEC
# ----------------------------------------------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

# Freeze CNN feature extractor
for param in wav2vec.feature_extractor.parameters():
    param.requires_grad = False

# Freeze first 10 transformer layers
for layer in wav2vec.encoder.layers[:10]:
    for param in layer.parameters():
        param.requires_grad = False

# Only last 2 transformer layers trainable
for layer in wav2vec.encoder.layers[10:]:
    for param in layer.parameters():
        param.requires_grad = True
# ----------------------------------------------------------
# DATASET CLASS
# ----------------------------------------------------------
def mean_pool_wav2vec(hidden):

    # average across time
    pooled = hidden.mean(dim=0)

    # L2 normalization (VERY IMPORTANT)
    pooled = pooled / (pooled.norm(p=2) + 1e-9)

    return pooled
class NativityDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio = preprocess_audio(row["audio_url"])

        inputs = processor(audio, sampling_rate=16000,
                           return_tensors="pt", padding=True)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        
        outputs = wav2vec(**inputs)

        hidden = outputs.last_hidden_state.squeeze(0)

# ---- POOLING (CRITICAL FIX) ----
        wav_emb = mean_pool_wav2vec(hidden)
        mfcc = torch.tensor(extract_mfcc(audio), dtype=torch.float32)

# force it to be a flat vector
        mfcc = mfcc.view(-1)

        mfcc = mfcc.to(device)

        lang_id = torch.tensor(row["lang_id"], dtype=torch.long).to(device)

        label = torch.tensor(row["nativity_status"], dtype=torch.float32).to(device)

        return wav_emb, mfcc, lang_id, label

# ----------------------------------------------------------
# ATTENTION POOLING
# ----------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=0)
        return torch.sum(weights * x, dim=0)



# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------
class NativityModel(nn.Module):
    def __init__(self, lang_count):
        super().__init__()

       
        self.lang_embed = nn.Embedding(lang_count, 8)

        wav_dim = 768
        mfcc_dim = 39
        lang_dim = 8

        input_dim = wav_dim + mfcc_dim + lang_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, wav_emb, mfcc, lang_id):
        wav_vec = wav_emb
        lang_vec = self.lang_embed(lang_id)

        combined = torch.cat([wav_vec, mfcc, lang_vec], dim=-1)
        x = self.dropout(self.relu(self.fc1(combined)))
        return self.fc2(x)


# ----------------------------------------------------------
# TRAINING
# ----------------------------------------------------------
train_loader = DataLoader(NativityDataset(train_df), batch_size=1, shuffle=True)
test_loader = DataLoader(NativityDataset(test_df), batch_size=1,shuffle=False)

model = NativityModel(len(lang_encoder.classes_)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW([
    {"params": model.parameters(), "lr": 2e-4},
    {"params": filter(lambda p: p.requires_grad, wav2vec.parameters()), "lr": 1e-5}
])

print("\nTraining started...\n")

for epoch in range(10):

    model.train()
    total_loss = 0

    for wav_emb, mfcc, lang_id, label in tqdm(train_loader):

        # ensure device consistency
        wav_emb = wav_emb.to(device)
        mfcc = mfcc.to(device)
        lang_id = lang_id.to(device)
        label = label.to(device)

        # forward
        logits = model(wav_emb, mfcc, lang_id)

        # compute loss
        loss = criterion(logits.view(-1), label.view(-1))

        # backward
        loss.backward()

        # prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update weights
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")
    # ================= EVALUATION =================
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for wav_emb, mfcc, lang_id, label in test_loader:

            wav_emb = wav_emb.to(device)
            mfcc = mfcc.to(device)
            lang_id = lang_id.to(device)
            label = label.to(device)

            logits = model(wav_emb, mfcc, lang_id)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds.view(-1) == label.view(-1)).sum().item()
            total += label.size(0)

    accuracy = correct / total
  

# ----------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------
print("\nEvaluating model...\n")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for wav_emb, mfcc, lang_id, label in test_loader:
        logits = model(wav_emb, mfcc, lang_id)
        probs = torch.softmax(logits, dim=-1)

        pred = torch.argmax(probs).item()

        if pred == label.item():
            correct += 1
        total += 1

print("FINAL ACCURACY:", correct/total)
# print("Mean probability:", probs.mean().item())
print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")
# ----------------------------------------------------------
# PREDICTION FUNCTION
# ----------------------------------------------------------
print(f"Epoch {epoch+1} Test Accuracy: {accuracy:.4f}")
def predict_audio(path, language):
    audio = preprocess_audio(path)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = wav2vec(**inputs)

    wav_emb = outputs.last_hidden_state.squeeze(0)
    mfcc = torch.tensor(extract_mfcc(audio), dtype=torch.float32).to(device)

    lang_id = torch.tensor(lang_encoder.transform([language])[0]).to(device)

    logits = model(wav_emb, mfcc, lang_id)
    probs = torch.softmax(logits, dim=-1)

    print("\nPrediction:", "Native" if torch.argmax(probs)==1 else "Non-Native")
    print("Confidence:", torch.max(probs).item())

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    import pandas as pd
    import numpy as np
    print("\n================ FINAL EVALUATION ================\n")

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_languages = []
    all_paths = []

    with torch.no_grad():
        for idx, (wav_emb, mfcc, lang_id, label) in enumerate(test_loader):

            wav_emb = wav_emb.to(device)
            mfcc = mfcc.to(device)
            lang_id = lang_id.to(device)
            label = label.to(device)

            logits = model(wav_emb, mfcc, lang_id)

        # ---- Confidence score ----
            prob = torch.sigmoid(logits).item()

        # ---- Predicted class ----
            pred = 1 if prob >= 0.5 else 0

        # Save
            all_labels.append(label.item())
            all_preds.append(pred)
            all_probs.append(prob)

        # Save metadata
            row = test_df.iloc[idx]
            all_languages.append(row["language"])
            all_paths.append(row["audio_url"])

# ================= METRICS =================

    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall    = recall_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=["Non-Native","Native"]))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))