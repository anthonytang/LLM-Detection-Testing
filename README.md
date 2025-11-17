# Gemma SIQA Evaluation (using Log-Likelihood Scoring)

This script evaluates the **Gemma-2-2B** model on the **Social IQa (SIQA)** dataset using full log-likelihood scoring across three answer choices (A, B, C), matching the published benchmark setup (~46-50 % accuracy).

---

### Overview
- Loads the SIQA dataset (`siqa_dataset.json`)
- Computes total log-likelihood for each answer choice given the context and question
- Selects the answer with the highest likelihood as the model prediction
- Iterates until **2000 correct** and **2000 incorrect** predictions are collected
- Saves detailed outputs for each example, including:
  - `context`, `question`, `answers` (A/B/C)
  - `gold` (true label)
  - `pred` (model prediction)
  - `loglikelihoods` (per-choice scores)
  - `is_correct` (True/False)
- Tracks total examples processed and overall model accuracy

---

### Output Files
- **`gemma_siqa_2000_each.json`**
  - `"correct"`: 2000 correctly predicted examples  
  - `"incorrect"`: 2000 incorrectly predicted examples  
  - `"summary"`: total processed count and collection stats

---

### Results
The full 2000 / 2000 collection maintains balanced subsets for interpretability analysis.
Gemma achieved 43.15% accuracy on 4635 randomly sampled questions from SIQA.
