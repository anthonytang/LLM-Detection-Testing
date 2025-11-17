# LLM Detection & Failure Mode Analysis

This repository contains a comprehensive suite of tools designed to analyze, predict, and understand the failure modes of Large Language Models (specifically **Gemma-2-2B**). The project uses a two-pronged approach:
1. **Behavioral Analysis:** Generating and testing natural language hypotheses about why the model fails.
2. **Mechanistic Interpretability:** Extracting and analyzing internal Sparse Autoencoder (SAE) features to understand the neural circuitry behind specific answers.

## Supported Datasets
The framework currently supports the following reasoning datasets:
- **BoolQ** (Boolean Question Answering)
- **Winogrande** (Commonsense Reasoning)
- **PIQA** (Physical Commonsense Reasoning)

## Project Structure

### 1. Behavioral Analysis Pipeline
These scripts use a "teacher" model (GPT-4o) to analyze the "student" model's (Gemma-2-2B) training data, generate hypotheses about its weaknesses, and predict future errors.

* **`process_all_datasets.py`**: The main entry point for behavioral analysis. It automates the entire pipeline for all supported datasets.
* **`dataset_processor.py`**: The core logic script. It handles:
    * Splitting data into Train/Val/Test sets.
    * Generating failure hypotheses using OpenAI's API.
    * Evaluating the model's predictions against these hypotheses.
    * Calculating accuracy metrics.

### 2. Mechanistic Interpretability (SAE)
These tools dive into the model's internals using Sparse Autoencoders to extract interpretable features from specific layers during inference.

* **`run_sae_feature_extraction.py`**: Runs inference on specific examples (like PIQA) using Gemma-2-2B. It hooks into the model to extract top-k activating features from JumpReLU SAEs across various layers.
* **`analyze_features.py`**: Enriches the raw SAE feature data by fetching human-readable explanations for features from **Neuronpedia**.

## Installation

Ensure you have Python installed along with the following dependencies:

```bash
pip install torch transformers openai huggingface_hub neuronpedia numpy tqdm
