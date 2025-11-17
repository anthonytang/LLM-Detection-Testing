"""
SIQA Probing Pipeline (Final Version)

This script trains linear probes on intermediate activations from the Gemma-2-2B model
to analyze representations on the Social IQa (SIQA) dataset. It performs last-token and
mean-token probing across selected layers and saves model weights, evaluation metrics,
and probability outputs for further analysis.

Features:
- Compatible with Gemma-2-2B via TransformerLens
- Efficient single-layer activation extraction
- Logistic regression probes (C=100)
- Saves metrics, classifier, and full probability outputs
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

FILES_DIR = "siqa_files"
OUT_DIR = "siqa_probes"
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(FILES_DIR, "siqa_train_split.json")
VAL_FILE = os.path.join(FILES_DIR, "siqa_val_split.json")
TEST_FILE = os.path.join(FILES_DIR, "siqa_test_split.json")

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
    """Load a dataset split (train/val/test) from JSON."""
    with open(path) as f:
        return json.load(f)

def build_prompts(items):
    """Construct SIQA-style text prompts for model input."""
    return [
        f"Context: {x['context']}\n"
        f"Question: {x['question']}\n"
        f"Answer A: {x['answerA']}\n"
        f"Answer B: {x['answerB']}\n"
        f"Answer C: {x['answerC']}\n"
        f"Correct answer: {x['label']}"
        for x in items
    ]

def get_cache(toks, layer_idx):
    """Extract resid_post activations for a specific layer."""
    layer_name = f"blocks.{layer_idx - 1}.hook_resid_post"
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n == layer_name)
    acts = cache[layer_name].detach().cpu()
    del cache
    torch.cuda.empty_cache()
    gc.collect()
    return acts

def collect_acts(items, labels, layer_idx, mode):
    """Collect activations for a given layer and pooling mode."""
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
    """Load a SIQA split and create binary labels (1 = correct, 0 = incorrect)."""
    d = load_split(path)
    items = d["correct"] + d["incorrect"]
    labels = [1] * len(d["correct"]) + [0] * len(d["incorrect"])
    return items, labels

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
train_items, y_train = prepare_data(TRAIN_FILE)
val_items, y_val = prepare_data(VAL_FILE)
test_items, y_test = prepare_data(TEST_FILE)

print("Train label counts:", np.bincount(y_train))
print("Validation label counts:", np.bincount(y_val))
print("Test label counts:", np.bincount(y_test))

# ---------------------------------------------------------------------
# Probing pipeline
# ---------------------------------------------------------------------
def run_probe(mode, train_items, y_train, val_items, y_val, test_items, y_test):
    """Run the probing process for a specified mode (last or mean)."""
    print(f"\n=== Running {mode} probe on SIQA ===")
    with tqdm(total=len(LAYERS), desc=f"{mode}_probe_overall", ncols=120) as outer:
        for L in LAYERS:
            start = time.time()
            print(f"\n--- Layer {L} ---")

            X_train, y_train = collect_acts(train_items, y_train, L, mode)
            X_val, y_val = collect_acts(val_items, y_val, L, mode)
            X_test, y_test = collect_acts(test_items, y_test, L, mode)

            # Train logistic regression probe
            clf = LogisticRegression(max_iter=1000, C=100)
            clf.fit(X_train, y_train)

            metrics = {
                "layer": L,
                "probe_type": mode,
                "train_acc": accuracy_score(y_train, clf.predict(X_train)),
                "val_acc": accuracy_score(y_val, clf.predict(X_val)),
                "test_acc": accuracy_score(y_test, clf.predict(X_test)),
                "train_logloss": log_loss(y_train, clf.predict_proba(X_train)),
                "val_logloss": log_loss(y_val, clf.predict_proba(X_val)),
                "test_logloss": log_loss(y_test, clf.predict_proba(X_test))
            }

            print(json.dumps(metrics, indent=2))

            # Save model weights and metrics
            np.savez(
                f"{OUT_DIR}/layer{L}_siqa_{mode}_weights.npz",
                weights=clf.coef_,
                bias=clf.intercept_
            )
            json.dump(
                metrics,
                open(f"{OUT_DIR}/layer{L}_siqa_{mode}_metrics.json", "w"),
                indent=2
            )

            # Save full probability outputs
            train_probs_full = clf.predict_proba(X_train)
            val_probs_full = clf.predict_proba(X_val)
            test_probs_full = clf.predict_proba(X_test)

            preds = {
                "train_probs_full": train_probs_full,
                "val_probs_full": val_probs_full,
                "test_probs_full": test_probs_full,
                "train_probs": train_probs_full[:, 1],
                "val_probs": val_probs_full[:, 1],
                "test_probs": test_probs_full[:, 1],
                "train_true": y_train,
                "val_true": y_val,
                "test_true": y_test,
                "train_items": train_items,
                "val_items": val_items,
                "test_items": test_items,
                "clf": clf,
                "metrics": metrics
            }

            joblib.dump(preds, f"{OUT_DIR}/layer{L}_siqa_{mode}_predictions.joblib")

            elapsed = time.time() - start
            outer.set_postfix_str(f"Layer {L} done ({elapsed / 60:.1f} min)")
            outer.update(1)
            torch.cuda.empty_cache()
            gc.collect()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_probe("last", train_items, y_train, val_items, y_val, test_items, y_test)
    run_probe("mean", train_items, y_train, val_items, y_val, test_items, y_test)
