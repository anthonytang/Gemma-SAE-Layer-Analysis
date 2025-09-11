import os
import json
import sys
import traceback
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download, login, list_repo_files
import random
import io
import zipfile
import requests
import argparse

def hf_login():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        print("Successfully logged into Hugging Face.")
    else:
        print("HF_TOKEN environment variable not set. Model download may fail.")
        print("Please set the HF_TOKEN environment variable with your token.")
        sys.exit(1)

def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32

def prepare_piqa_files():
    if os.path.exists("dev.jsonl") and os.path.exists("dev-labels.lst"):
        print("Found dev.jsonl and dev-labels.lst — skipping dataset download.")
        return

    url = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
    print("Downloading PIQA dataset from:", url)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PIQA dataset: {e}")
        sys.exit(1)

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        src_jsonl = "physicaliqa-train-dev/dev.jsonl"
        src_labels = "physicaliqa-train-dev/dev-labels.lst"
        try:
            with zf.open(src_jsonl) as fin, open("dev.jsonl", "wb") as fout:
                fout.write(fin.read())
            with zf.open(src_labels) as fin, open("dev-labels.lst", "wb") as fout:
                fout.write(fin.read())
        except KeyError as e:
            print(f"Error extracting files from zip. Expected file not found: {e}")
            sys.exit(1)

    n_json = sum(1 for _ in open("dev.jsonl", "r", encoding="utf-8"))
    n_lbls = sum(1 for _ in open("dev-labels.lst", "r", encoding="utf-8"))
    print(f"Wrote dev.jsonl ({n_json} lines) and dev-labels.lst ({n_lbls} lines).")

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        return self.decode(acts)

@torch.no_grad()
def get_solution_details(model, tokenizer, question, solution, device, all_saes, top_k):
    full_prompt = f"Question: {question}\nSolution: {solution}"
    question_tokens = tokenizer.encode(f"Question: {question}\nSolution: ", return_tensors="pt").to(device)
    full_tokens = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    if full_tokens.shape[1] <= question_tokens.shape[1]:
        return -float('inf'), {}

    captured_activations = {}
    def create_hook(layer_idx):
        def gather_hook(mod, inputs, outputs):
            captured_activations[layer_idx] = (outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
        return gather_hook

    hook_handles = []
    for layer_idx in all_saes.keys():
        handle = model.model.layers[layer_idx].register_forward_hook(create_hook(layer_idx))
        hook_handles.append(handle)

    outputs = model(full_tokens)

    for handle in hook_handles:
        handle.remove()

    shifted_logits = outputs.logits[..., :-1, :].contiguous()
    shifted_labels = full_tokens[..., 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    log_likelihood_per_token = log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)

    solution_start_idx = question_tokens.shape[1] - 1
    solution_log_likelihood = log_likelihood_per_token[0, solution_start_idx:].sum().item()

    all_feature_details = {}
    for layer_idx, activation_tensor in captured_activations.items():
        try:
            sae = all_saes[layer_idx]
            last_token_act = activation_tensor[0, -1, :].to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            sae_activations = sae.encode(last_token_act).squeeze()

            k = min(top_k, sae_activations.numel())
            top_k_activations, top_k_indices = torch.topk(sae_activations, k)

            layer_feature_details = []
            for act, idx in zip(top_k_activations, top_k_indices):
                if act.item() > 0:
                    layer_feature_details.append({
                        "feature_id": idx.item(),
                        "activation": act.item(),
                        "layer": layer_idx
                    })
            
            if layer_feature_details:
                all_feature_details[str(layer_idx)] = layer_feature_details

        except Exception as e:
            print(f"Error processing SAE activations for layer {layer_idx}: {e}")
            traceback.print_exc()

    return solution_log_likelihood, all_feature_details

@torch.no_grad()
def evaluate_piqa_example(model, tokenizer, device, all_saes, top_k, example):
    question = example["goal"]
    sol1, sol2 = example["sol1"], example["sol2"]
    correct_label = int(example["label"])

    ll_s1, features_s1 = get_solution_details(model, tokenizer, question, sol1, device, all_saes, top_k)
    ll_s2, features_s2 = get_solution_details(model, tokenizer, question, sol2, device, all_saes, top_k)

    predicted_label = 0 if ll_s1 >= ll_s2 else 1
    is_correct = (predicted_label == correct_label)

    if correct_label == 0:
        positive_details = {"text": sol1, "log_likelihood": ll_s1, "active_features_by_layer": features_s1}
        negative_details = {"text": sol2, "log_likelihood": ll_s2, "active_features_by_layer": features_s2}
    else:
        positive_details = {"text": sol2, "log_likelihood": ll_s2, "active_features_by_layer": features_s2}
        negative_details = {"text": sol1, "log_likelihood": ll_s1, "active_features_by_layer": features_s1}
    
    pos_feature_count = sum(len(v) for v in positive_details['active_features_by_layer'].values())
    neg_feature_count = sum(len(v) for v in negative_details['active_features_by_layer'].values())

    print(f"\n--- Evaluating Example ID: {example['id']} ---")
    print(f"Question: {question}")
    print(f"Positive Solution ('{positive_details['text']}'): LL={positive_details['log_likelihood']:.2f}, Total Top Features={pos_feature_count}")
    print(f"Negative Solution ('{negative_details['text']}'): LL={negative_details['log_likelihood']:.2f}, Total Top Features={neg_feature_count}")
    print(f"Prediction was {'CORRECT' if is_correct else 'INCORRECT'}")

    return {
        "example_id": example["id"],
        "question": question,
        "model_is_correct": bool(is_correct),
        "positive_solution_details": positive_details,
        "negative_solution_details": negative_details,
    }

def main(args):
    hf_login()
    prepare_piqa_files()

    device, dtype = get_device_and_dtype()
    print(f"\nUsing device: {device} | dtype: {dtype}")

    print("Loading Gemma-2-2B model and tokenizer...")
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        device_map='auto',
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded.")

    print("\nDownloading and loading all available pre-trained SAEs...")
    sae_repo_id = "google/gemma-scope-2b-pt-res"
    all_saes = {}

    print(f"Scanning repository '{sae_repo_id}' for SAE files...")
    try:
        repo_files = list_repo_files(repo_id=sae_repo_id)
    except Exception as e:
        print(f"FATAL: Could not list files in repository. Check network/token. Error: {e}")
        sys.exit(1)

    for layer_idx in range(29):
        prefix = f"layer_{layer_idx}/"
        sae_filename = next((f for f in repo_files if f.startswith(prefix) and f.endswith("/params.npz")), None)

        if sae_filename:
            print(f"Found SAE for layer {layer_idx} at '{sae_filename}'. Downloading...")
            try:
                path_to_params = hf_hub_download(repo_id=sae_repo_id, filename=sae_filename)
                params = np.load(path_to_params)
                d_model, d_sae = params["W_enc"].shape
                sae = JumpReLUSAE(d_model, d_sae).to(device)
                pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
                sae.load_state_dict(pt_params, strict=True)
                sae.eval()
                all_saes[layer_idx] = sae
                print(f"Successfully loaded SAE for layer {layer_idx}")
            except Exception as e:
                print(f"WARNING: Found but could not load SAE for layer {layer_idx}. Error: {e}")
        else:
            print(f"SAE for layer {layer_idx} not found on Hub. Skipping.")
    
    if not all_saes:
        print("\nFATAL: No SAEs could be loaded. Please check network connection and token permissions.")
        sys.exit(1)
    print(f"\nLoaded a total of {len(all_saes)} SAEs.")

    try:
        with open("dev.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        with open("dev-labels.lst", "r", encoding="utf-8") as f:
            labels = [int(l.strip()) for l in f]

        piqa_examples = []
        for i, (ex, lab) in enumerate(zip(data, labels)):
            ex['id'] = i
            ex['label'] = lab
            piqa_examples.append(ex)
        print(f"\nLoaded {len(piqa_examples)} PIQA dev examples.")
    except Exception as e:
        print(f"Failed to read PIQA dev files: {e}")
        sys.exit(1)

    if len(piqa_examples) < args.num_samples:
        print(f"Warning: Dataset has fewer than {args.num_samples} examples. Using all {len(piqa_examples)}.")
        sampled_examples = piqa_examples
    else:
        sampled_examples = random.sample(piqa_examples, args.num_samples)

    print(f"\nEvaluating {len(sampled_examples)} randomly sampled PIQA examples...")
    print(f"Extracting top {args.top_k} features from {len(all_saes)} layers for each solution.")

    results = []
    correct_count = 0
    total_count = len(sampled_examples)
    for i, example in enumerate(sampled_examples):
        try:
            res = evaluate_piqa_example(model, tokenizer, device, all_saes, args.top_k, example)
            results.append(res)
            if res["model_is_correct"]:
                correct_count += 1

            accuracy = (correct_count / (i + 1)) * 100
            print(f"Progress: {i+1}/{total_count} | Correct: {correct_count} | Accuracy: {accuracy:.2f}%")
        except Exception as e:
            print(f"An error occurred while evaluating example {example.get('id', i)}: {e}")
            traceback.print_exc()
            results.append({"example_id": example.get('id', f"index_{i}"), "error": str(e)})

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    final_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("\n" + "="*50)
    print("Evaluation Complete")
    print("="*50)
    print(f"Total Examples Evaluated: {total_count}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Detailed results saved to {args.output_file}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemma-2-2B with SAE on PIQA dataset across all layers.")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-file", type=str, default="piqa_sae_top100_all_layers.json")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    main(args)