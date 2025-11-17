# Gemma BoolQ Evaluation

This repository runs the **Gemma-2-2B** model on a randomly sampled subset of the **BoolQ** dataset to evaluate its question-answering performance.

## Overview
- **Goal:** Collect the first 2000 correct and 2000 incorrect BoolQ responses from Gemma.  
- **Model:** Gemma-2-2B  
- **Dataset:** BoolQ (Boolean question-answer pairs)

## Files
- **`boolq_dataset.json`** — Contains ~12.5k randomly sampled BoolQ questions.  
- **`gemma_boolq_answers.json`** — Stores Gemma’s answers, capped at 2000 correct and 2000 incorrect.  
- **`run_gemma_boolq.py`** — Main script that runs inference and saves results.

## Usage
1. Make sure the Gemma model is already downloaded locally.  
2. Place `boolq_dataset.json` in the same directory.  
3. Run:
   ```bash
   python run_gemma_boolq.py
