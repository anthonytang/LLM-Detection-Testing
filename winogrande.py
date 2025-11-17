"""
Winogrande Probing Pipeline (Final Version)

This script trains linear probes on intermediate activations from the Gemma-2-2B model
to analyze representations on the Winogrande dataset. It performs both last-token and
mean-token probing across selected layers, saving model weights, evaluation metrics,
and probability outputs for analysis.

Features:
- Extracts residual activations via TransformerLens
- Supports last-token and mean-token probes
- Uses logistic regression with C=100
- Saves per-layer weights, metrics, and probability outputs
"""

import os
import json
import gc
import time
import torch
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from huggingface_hub import login, HfApi
from transformer_lens import HookedTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------
# Configuration and authentication
# ---------------------------------------------------------------------
HF_TOKEN = input("Enter your Hugging Face token (starts with hf_...): ").strip()

FILES_DIR = "winogrande_files"
OUT_DIR = "winogrande_probes"
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(FILES_DIR, "gemma_winogrande_2000_each_train_split.json")
VAL_FILE = os.path.join(FILES_DIR, "gemma_winogrande_2000_each_val_split.json")
TEST_FILE = os.path.join(FILES_DIR, "gemma_winogrande_2000_each_test_split.json")

LAYERS = [5, 10, 15, 20, 25]
MAX_LEN = 256
BATCH_SZ = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# Authenticate with Hugging Face
# ---------------------------------------------------------------------
print("Authenticating with Hugging Face...")
login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()
model_id = "google/gemma-2-2b"
api.model_info(model_id, token=HF_TOKEN)
print(f"Access confirmed for {model_id}")

torch.cuda.empty_cache()
gc.collect()

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
print("Loading Gemma-2-2B (bfloat16 safe mode)...")
model = HookedTransformer.from_pretrained(
    model_id,
    device="cpu",
    dtype=torch.bfloat16
)
model.to(DEVICE, torch.bfloat16)
model.cfg.use_attn_result = False
model.cfg.use_split_qkv_input = False
print("Model loaded successfully on", DEVICE)

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_split(path):
    """Load dataset split (train/val/test) from JSON."""
    with open(path) as f:
        d = json.load(f)
    return d["correct"] + d["incorrect"]

def build_prompts(items):
    """Construct Winogrande-style prompts for the model."""
    return [
        f"Question: {x['question']}\nOption 1: {x['option1']}\n"
        f"Option 2: {x['option2']}\nAnswer (1 or 2): "
        for x in items
    ]

def get_cache(toks, layer_idx):
    """Return resid_post activations for a specific layer."""
    layer_name = f"blocks.{layer_idx - 1}.hook_resid_post"
    _, cache = model.run_with_cache(toks, names_filter=lambda name: name == layer_name)
    acts = cache[layer_name].detach().cpu()
    del cache
    torch.cuda.empty_cache()
    gc.collect()
    return acts

def collect_acts(items, labels, layer_idx, mode):
    """Extract activations for a given layer and pooling mode."""
    vecs = []
    with tqdm(total=len(items), desc=f"layer{layer_idx}_{mode}", ncols=100) as pbar:
        for start in range(0, len(items), BATCH_SZ):
            batch = items[start:start + BATCH_SZ]
            toks = model.to_tokens(build_prompts(batch), prepend_bos=True).to(DEVICE)
            if toks.shape[1] > MAX_LEN:
                toks = toks[:, :MAX_LEN]
            acts = get_cache(toks, layer_idx)
            if mode == "last":
                v = acts[:, -1, :].float().cpu().numpy()
            else:
                v = acts.mean(dim=1).float().cpu().numpy()
            vecs.append(v)
            pbar.update(len(batch))
            torch.cuda.empty_cache()
            gc.collect()
    X = np.concatenate(vecs, axis=0)
    y = np.array(labels, dtype=int)
    return X, y

def prepare_data(path):
    """Load dataset and extract binary labels (1 = correct, 0 = incorrect)."""
    items = load_split(path)
    labels = [int(x["is_correct"]) for x in items]
    return items, labels

train_items, y_train = prepare_data(TRAIN_FILE)
val_items, y_val = prepare_data(VAL_FILE)
test_items, y_test = prepare_data(TEST_FILE)

# ---------------------------------------------------------------------
# Probing pipeline
# ---------------------------------------------------------------------
def run_probe(mode, train_items, y_train, val_items, y_val, test_items, y_test):
    """Run the probing process for a specified mode (last or mean)."""
    print(f"\n=== Running {mode} probe on Winogrande (C=100) ===")
    with tqdm(total=len(LAYERS), desc=f"{mode}_probe_overall", ncols=120) as outer:
        for L in LAYERS:
            start = time.time()
            print(f"\n--- Layer {L} ---")

            # Extract activations
            X_train, y_train = collect_acts(train_items, y_train, L, mode)
            X_val, y_val = collect_acts(val_items, y_val, L, mode)
            X_test, y_test = collect_acts(test_items, y_test, L, mode)

            # Train logistic regression probe
            clf = LogisticRegression(C=100, max_iter=1000)
            clf.fit(X_train, y_train)

            # Generate predictions and probabilities
            y_train_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            y_test_pred = clf.predict(X_test)

            y_train_prob = clf.predict_proba(X_train)
            y_val_prob = clf.predict_proba(X_val)
            y_test_prob = clf.predict_proba(X_test)

            # Compute metrics
            metrics = {
                "layer": L,
                "probe_type": mode,
                "C": 100,
                "train_acc": accuracy_score(y_train, y_train_pred),
                "val_acc": accuracy_score(y_val, y_val_pred),
                "test_acc": accuracy_score(y_test, y_test_pred),
                "train_logloss": log_loss(y_train, y_train_prob),
                "val_logloss": log_loss(y_val, y_val_prob),
                "test_logloss": log_loss(y_test, y_test_prob)
            }

            # Save metrics
            json.dump(
                metrics,
                open(f"{OUT_DIR}/layer{L}_winogrande_{mode}_metrics.json", "w"),
                indent=2
            )

            # Save weights
            np.savez(
                f"{OUT_DIR}/layer{L}_winogrande_{mode}_weights.npz",
                weights=clf.coef_,
                bias=clf.intercept_
            )

            # Save predictions and probabilities
            preds = {
                "train": {"y_true": y_train, "y_pred": y_train_pred, "y_prob": y_train_prob},
                "val": {"y_true": y_val, "y_pred": y_val_pred, "y_prob": y_val_prob},
                "test": {"y_true": y_test, "y_pred": y_test_pred, "y_prob": y_test_prob}
            }
            joblib.dump(preds, f"{OUT_DIR}/layer{L}_winogrande_{mode}_predictions.joblib")

            print(json.dumps(metrics, indent=2))
            print(f"Saved predictions → {OUT_DIR}/layer{L}_winogrande_{mode}_predictions.joblib")

            outer.set_postfix_str(f"Layer {L} done")
            outer.update(1)
            torch.cuda.empty_cache()
            gc.collect()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_probe("last", train_items, y_train, val_items, y_val, test_items, y_test)
    run_probe("mean", train_items, y_train, val_items, y_val, test_items, y_test)
