# Gemma WinoGrande Evaluation (using Partial Log-Likelihood)

This script evaluates the **Gemma-2-2B** model on the **WinoGrande** dataset using **partial log-likelihood scoring**, matching the official benchmark setup (~70 % accuracy).

## Overview
- Loads the **WinoGrande** dataset (`winogrande_train.json`)
- Uses **partial log-likelihood** (prefix + option → predict suffix) instead of prompting
- Iterates until **2000 correct** and **2000 incorrect** predictions are collected
- Saves detailed outputs for each example, including:
  - `question`, `option1`, `option2`
  - `gold` (true label)
  - `gemma_answer` (model prediction)
  - `ll_option1`, `ll_option2` (log-likelihoods)
  - `is_correct` (True/False)
- Tracks total examples processed and overall accuracy

## Output Files
- **`gemma_winogrande_2000_each.json`**
  - Contains:
    - `"correct"`: 2000 correctly predicted examples  
    - `"incorrect"`: 2000 incorrectly predicted examples  
    - `"summary"`: total processed count and collection stats

## Results
- Gemma-2-2b achieved **69.87%** accuracy on **5725 randomly sampled questions** from winogrande_xl dataset's train split.
