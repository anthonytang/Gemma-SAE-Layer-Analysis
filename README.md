**Gemma-SAE-Layer-Analysis**

This repository contains a Python script for analyzing the internal feature activations of Google's Gemma-2-2B model. It uses the Physical IQa (PIQA) commonsense reasoning dataset to evaluate the model's choices and leverages pre-trained Sparse Autoencoders (SAEs) from the GemmaScope project to interpret the model's behavior across all available neural network layers.

**Project Goal**

The primary goal of this project is to "look inside the mind" of a large language model. When Gemma-2-2B evaluates two possible solutions to a problem, what internal "features" or "concepts" become active? Do different features fire for the correct answer versus the incorrect one? How do these feature activations change across the depth of the network (from early to late layers)?

This script provides the tool to gather this data, forming a foundation for mechanistic interpretability research.

**How It Works**

For each question in the PIQA dataset, the script performs the following steps:

**Model & SAE Loading:** It loads the pre-trained Gemma-2-2B model and tokenizer. It then dynamically scans the official google/gemma-scope-2b-pt-res Hugging Face repository to find and load all available SAEs for each layer of the model.

**Log-Likelihood Calculation:** For a given question, it evaluates both the correct (positive) and incorrect (negative) solutions, calculating the log-likelihood score for each to determine the model's own prediction.

**Activation Capture:** In a single, efficient forward pass, it uses PyTorch hooks to capture the residual stream activations from every layer that has a corresponding SAE. This is done for the final token of each solution.

**Feature Extraction:** The captured activations for each layer are passed through their respective SAEs. The SAE encoder output reveals the activation strengths of thousands of learned, interpretable features.

**Data Aggregation:** The script identifies the top K features (default is 100) with the highest activation scores for each layer and for each solution.

**Output Generation:** The results are compiled into a structured JSON file. Each entry contains the question, the positive and negative solutions, their log-likelihoods, and a detailed breakdown of the top-activating features by layer.

**Setup & Prerequisites**
1. Clone the Repository

2. Install Dependencies
You can install them using pip:

```pip install torch transformers huggingface_hub numpy requests```

3. Hugging Face Authentication

To download the Gemma model and the SAEs, you need a Hugging Face account with access granted to Gemma.

Make sure you have accepted the Gemma license on the Hugging Face model page.

Create an access token with "read" permissions in your Hugging Face account settings.

Set this token as an environment variable.

On macOS/Linux:

```export HF_TOKEN="your_token_here"```

On Windows (Command Prompt):

```set HF_TOKEN="your_token_here"```

The script will use this environment variable to authenticate automatically.

**Usage**

The script can be run from the command line.

```python run_sae_feature_extraction_all_layers.py```

**Command-Line Arguments**

You can customize the script's behavior with the following arguments:

--num-samples: The number of random examples to evaluate from the PIQA dataset. (Default: 50)

--output-file: The name of the JSON file to save the results to. (Default: piqa_sae_top100_all_layers.json)

--top-k: The number of top-activating features to record for each layer. (Default: 100)

Example: To run on 200 samples and save the top 50 features to a file named results.json:

```python run_sae_feature_extraction_all_layers.py --num-samples 200 --top-k 50 --output-file results.json```

**Output Format**

The script generates a JSON file containing a list of objects, where each object represents one evaluated PIQA example. Here is the structure of a single entry:

```
[
  {
    "example_id": 1714,
    "question": "soap",
    "model_is_correct": true,
    "positive_solution_details": {
      "text": "can fit in basket ",
      "log_likelihood": -24.12,
      "active_features_by_layer": {
        "8": [
          {
            "feature_id": 1234,
            "activation": 5.81,
            "layer": 8
          }
        ],
        "20": [
          {
            "feature_id": 5678,
            "activation": 7.23,
            "layer": 20
          }
        ]
      }
    },
    "negative_solution_details": {
      "text": "can fit in baking sheet ",
      "log_likelihood": -32.19,
      "active_features_by_layer": {
        "8": [
          {
            "feature_id": 4321,
            "activation": 6.11,
            "layer": 8
          }
        ],
        "20": [
          {
            "feature_id": 8765,
            "activation": 4.98,
            "layer": 20
          }
        ]
      }
    }
  }
]
```
(Note: Feature lists are truncated for brevity)
