# GPT-2 From Scratch (Single Notebook Edition)

This folder contains the complete, consolidated implementation of the **GPT-2 architecture from scratch**, contained entirely within a single Jupyter notebook without any `%run` dependencies on other notebooks or external folders.

---

## Files in this Folder

| File | Description |
| :--- | :--- |
| [`GPT2_From_Scratch.ipynb`](./GPT2_From_Scratch.ipynb) | Master notebook containing the entire GPT-2 pipeline from tokenization to fine-tuning. |
| [`gpt_download.py`](./gpt_download.py) | Checkpoint loader utility with **automatic weight discovery** that reuses weights already downloaded in previous folders (`../3. LLM Architecture/gpt2` or `../4. Fine-Tuning/gpt2`). |
| [`the-verdict.txt`](./the-verdict.txt) | Sample short story text by Edith Wharton used for tokenizer demonstrations and pretraining. |

---

## Notebook Structure

The notebook is divided into 8 clean, modular sections:

1. **Environment & Configurations**: Library imports, device detection (`cuda`, `mps`, `cpu`), and model hyperparameter configurations for 124M and 355M GPT-2 models.
2. **Data Preprocessing & Input Pipeline**: Raw text loading, Byte Pair Encoding (BPE) via `tiktoken`, custom `GPTDatasetV1` with sliding window, and PyTorch `DataLoader` setup.
3. **Attention Mechanisms**: Complete from-scratch implementation of `MultiHeadAttention` with query/key/value projections, causal upper-triangular masking, scaled dot-product attention, and dropout.
4. **GPT-2 Architecture Building Blocks**:
   - `LayerNorm` (operating across embedding dimension with learnable scale and shift, `unbiased=False`).
   - `GELU` (tanh approximation formula).
   - `FeedForward` (4x expansion $\rightarrow$ GELU $\rightarrow$ contraction).
   - `TransformerBlock` (Pre-LayerNorm with residual connections).
   - `GPTModel` (embedding layers, transformer block stack, final layer norm, and linear output head).
5. **Text Generation & Loss Evaluation**:
   - Helper utilities: `text_to_token_ids` and `token_ids_to_text`.
   - Greedy text generation: `generate_text_simple`.
   - Advanced text generation: `generate` with temperature scaling and top-$k$ filtering.
   - Cross Entropy Loss (`calc_loss_batch`, `calc_loss_loader`) and Perplexity.
6. **Complete Training & Pretraining Loop**:
   - Model evaluation: `evaluate_model`.
   - Sample generation tracker: `generate_and_print_sample`.
   - Full training loop: `train_model_simple` with AdamW optimizer and loss recording.
   - Pretraining demo on `the-verdict.txt` and loss curve plotting (`plot_losses`).
   - Saving and reloading model checkpoints.
7. **Loading Pretrained OpenAI GPT-2 Weights**:
   - Reuses already-downloaded weights in `../3. LLM Architecture/gpt2` or `../4. Fine-Tuning/gpt2` without redownloading ~500MB of data.
   - Weight mapping functions: `assign` and `load_weights_into_gpt`.
   - Inference demo generating coherent paragraphs from text prompts.
8. **Fine-Tuning: Classification & SFT**:
   - Replacing the output head with a classification linear layer.
   - Freezing base transformer layers.
   - `SpamDataset`, accuracy evaluation (`calc_accuracy_loader`), and spam vs. ham prediction on custom text inputs.
   - Loading pre-trained fine-tuning weights if available.

---

## How to Run

Open [`GPT2_From_Scratch.ipynb`](./GPT2_From_Scratch.ipynb) in VS Code or JupyterLab and execute the cells sequentially from top to bottom.
