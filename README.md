# 🚀 GPT-2 From Scratch: Complete Architecture, Pretraining, Fine-Tuning & Web Deployment

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only%20Transformer-orange.svg)]()
[![Model Size](https://img.shields.io/badge/Models-124M%20%7C%20355M-purple.svg)]()

A comprehensive, ground-up implementation of OpenAI's **GPT-2 (Generative Pre-trained Transformer 2)** architecture in PyTorch without relying on high-level Hugging Face transformer wrappers. 

This repository documents the entire deep learning lifecycle for Large Language Models: from **byte-pair tokenization** and **causal multi-head attention**, to **pretraining**, **model weight transfer**, **classification fine-tuning (95.67% test accuracy)**, **instruction fine-tuning (Alpaca SFT)**, and **real-time web serving**.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture & Specifications](#-architecture--specifications)
- [Repository Structure](#-repository-structure)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
  - [1. Data Preprocessing & Input Pipeline](#1-data-preprocessing--input-pipeline)
  - [2. Attention Mechanisms Deep-Dive](#2-attention-mechanisms-deep-dive)
  - [3. Full Transformer Architecture](#3-full-transformer-architecture)
  - [4. Pretraining & Weight Transfer](#4-pretraining--weight-transfer)
  - [5. Fine-Tuning](#5-fine-tuning)
    - [A. Spam Classification Fine-Tuning](#a-spam-classification-fine-tuning-124m)
    - [B. Instruction Following (SFT)](#b-instruction-following-sft-355m)
  - [6. Text Generation & Decoding](#6-text-generation--decoding)
- [Interactive Web Application](#-interactive-web-application)
- [Getting Started](#-getting-started)
- [Results & Benchmarks](#-results--benchmarks)
- [License](#-license)

---

## 🌟 Project Overview

- **Built from First Principles:** Every module—LayerNorm, GELU activation, Multi-Head Causal Attention, Feed-Forward Networks, and Transformer Blocks—is implemented directly using fundamental PyTorch tensor operations.
- **Support for Multiple Model Scales:** Fully parameterizable for **GPT-2 Small (124M)** and **GPT-2 Medium (355M)** configurations.
- **OpenAI Checkpoint Loading:** Includes custom weight-mapping utilities to load pretrained OpenAI TensorFlow/PyTorch weights directly into the scratch architecture without external transformer libraries.
- **Multi-Task Downstream Adaptation:**
  - **Text Classification:** Replaced output head and trained a spam classifier achieving **95.67% test accuracy** and **97.32% validation accuracy** on the SMS Spam Collection dataset.
  - **Instruction Tuning (SFT):** Fine-tuned the 355M parameter model using Alpaca-style instruction/response formatting for conversational QA and task completion.
- **Interactive Full-Stack Web App:** A dual-engine (Flask + Python standard library fallback) web UI featuring real-time chat, spam detection, and creative generation with temperature and top-$k$ sampling sliders.

---

## 📐 Architecture & Specifications

The model follows the decoder-only transformer architecture introduced in *Language Models are Unsupervised Multitask Learners* (Radford et al., 2019):

```
Input Tokens ──> [Token Embedding + Learned Positional Embedding]
                        │
                        ▼
         ┌───────────────────────────────┐
         │       Transformer Block       │  x N Layers (12 / 24)
         │  ┌─────────────────────────┐  │
         │  │ Pre-LayerNorm           │  │
         │  │ Multi-Head Causal Attn  │  │
         │  │ Dropout + Residual Add  │  │
         │  ├─────────────────────────┤  │
         │  │ Pre-LayerNorm           │  │
         │  │ Feed-Forward (GELU)     │  │
         │  │ Dropout + Residual Add  │  │
         │  └─────────────────────────┘  │
         └──────────────┬────────────────┘
                        │
                        ▼
               [Final LayerNorm]
                        │
                        ▼
         [Linear Output Head / LM Head] ──> Logits (Vocab: 50,257)
```

### Hyperparameter Configurations

| Parameter | GPT-2 Small (124M) | GPT-2 Medium (355M) |
| :--- | :---: | :---: |
| **Vocabulary Size ($V$)** | 50,257 | 50,257 |
| **Context Window ($T$)** | 1,024 | 1,024 |
| **Embedding Dimension ($d_{model}$)** | 768 | 1,024 |
| **Transformer Layers ($n_{layer}$)** | 12 | 24 |
| **Attention Heads ($n_{head}$)** | 12 | 16 |
| **Head Dimension ($d_k$)** | 64 | 64 |
| **Feed-Forward Expansion** | $4 \times d_{model}$ (3,072) | $4 \times d_{model}$ (4,096) |
| **Activation Function** | GELU (tanh approx.) | GELU (tanh approx.) |
| **Dropout Rate** | 0.1 (0.0 for inference) | 0.1 (0.0 for inference) |

---

## 📂 Repository Structure

```
.
├── 1. Data Preprocessing/             # BPE Tokenization, embeddings & sliding window dataset
│   ├── 1. LLM Data Pre-processing.ipynb
│   ├── 2. Vector Embedding.ipynb
│   └── the-verdict.txt
├── 2. Attention Mechanisms/           # Evolution from simple attention to Multi-Head Causal Attention
│   ├── 1. Simple Attention Mechanism.ipynb
│   ├── 2. Self-Attention.ipynb
│   ├── 3. Causal Attention.ipynb
│   └── 4. Multi-Head Attention.ipynb
├── 3. LLM Architecture/               # Complete GPT-2 model, pretraining loop & weight loading
│   ├── 1. GPT model from scratch.ipynb
│   ├── 2. Evaluating LLMs.ipynb
│   ├── 3. Complete Training loop.ipynb
│   ├── gpt_download.py
│   └── sms_spam_collection/
├── 4. Fine-Tuning/                    # Classification fine-tuning & Instruction SFT
│   ├── 1. Fine-Tuning.ipynb
│   ├── instruction-data.json
│   ├── gpt2-small124M-classification-finetuning.pth
│   └── gpt2-medium355M-sft.pth
├── GPT2-Single-Notebook/              # Standalone, dependency-free master notebooks
│   ├── GPT2_From_Scratch.ipynb
│   ├── 1_GPT2_Classification_FineTuning.ipynb
│   └── 2_GPT2_Instruction_FineTuning.ipynb
├── web_app/                           # Full-stack interactive inference web interface
│   ├── app.py                         # Flask & ThreadingHTTPServer backend
│   ├── model_engine.py                # Unified PyTorch inference manager
│   ├── templates/index.html           # Responsive multi-tab UI
│   └── static/                        # Stylesheet and dynamic frontend scripts
└── README.md
```

---

## 🔬 Pipeline Walkthrough

### 1. Data Preprocessing & Input Pipeline
- **Byte-Pair Encoding (BPE):** Employs the official GPT-2 tokenizer via `tiktoken` with a vocabulary size of 50,257 tokens, preventing out-of-vocabulary issues.
- **Sliding-Window Dataset (`GPTDatasetV1`):** Chunks contiguous text into fixed-length sequences of length $T$, paired with target tokens shifted by 1 step for autoregressive next-token prediction.
- **Vector Embedding Layer:** Combines token embeddings ($W_{te} \in \mathbb{R}^{V \times d}$) with learned positional embeddings ($W_{pe} \in \mathbb{R}^{T \times d}$).

### 2. Attention Mechanisms Deep-Dive
Step-by-step evolution implemented across progressive notebooks:
1. **Simple Attention:** Unweighted dot-product attention between input vectors.
2. **Self-Attention with Trainable Weights:** Introduction of Query ($W_q$), Key ($W_k$), and Value ($W_v$) projections.
3. **Causal Masking:** Applying an upper-triangular mask with $-\infty$ above the diagonal before the softmax step to preserve the autoregressive invariant (tokens can only attend to prior positions):
   $$\text{Masked Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V, \quad M_{ij} = \begin{cases} 0 & i \ge j \\ -\infty & i < j \end{cases}$$
4. **Multi-Head Attention:** Splitting projected dimensions into multiple heads, executing parallel attention operations, concatenating results, and applying an output linear projection ($W_o$).

### 3. Full Transformer Architecture
- **Layer Normalization:** Custom `LayerNorm` with learnable scale ($\gamma$) and shift ($\beta$) parameters computed over the feature dimension with `unbiased=False`.
- **GELU Activation:** Implemented using the fast tanh approximation:
  $$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$$
- **Feed-Forward Sub-layer:** Two linear projections expanding the hidden state to $4 \times d_{model}$ before projecting back.
- **Pre-LayerNorm Residual Connection:** Normalization applied *before* attention and feed-forward operations, stabilizing gradient flow across deep networks.

### 4. Pretraining & Weight Transfer
- **Pretraining Loop:** Built training and evaluation loops utilizing cross-entropy loss and perplexity logging.
- **Pretrained OpenAI Weight Loading:** Custom tensor slicing and reshaping scripts (`gpt_download.py`) load official OpenAI checkpoint weights directly into the native PyTorch module parameters.

### 5. Fine-Tuning

#### A. Spam Classification Fine-Tuning (124M)
- The pretrained output token generation head is replaced with a 2-class linear classification head ($d_{model} \rightarrow 2$).
- Base transformer backbone is frozen initially, training the classification head on the SMS Spam Collection dataset before end-to-end tuning.
- **Results:**
  - **Test Accuracy:** **95.67%**
  - **Validation Accuracy:** **97.32%**
  - **Training Accuracy:** **97.21%**

#### B. Instruction Following (SFT) (355M)
- Applied Supervised Fine-Tuning (SFT) to the 355M parameter GPT-2 Medium model using the Alpaca instruction-response prompt structure:
  ```
  Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {instruction}

  ### Input:
  {optional_context}

  ### Response:
  {generated_response}
  ```
- Evaluated on prompt completion, multi-turn Q&A, and formatting tasks.

### 6. Text Generation & Decoding
The generation module (`generate`) supports multiple decoding strategies:
- **Greedy Decoding:** Taking $\text{argmax}$ over output logits.
- **Temperature Scaling ($T$):** Controls randomness by dividing logits prior to softmax ($z_i / T$).
- **Top-$k$ Sampling:** Truncates the vocabulary distribution to the top $k$ most probable tokens before sampling to prevent degeneration and repetitive loops.

---

## 💻 Interactive Web Application

The `web_app/` directory provides a clean, responsive interface to test and demonstrate all models:

<div align="center">
  <p><b>AI Assistant (Instruction SFT 355M) &bull; Spam Detector (Classifier 124M) &bull; Creative Generator (Base 124M)</b></p>
</div>

### Key Capabilities:
- **Zero-Dependency Fallback:** Runs seamlessly on standard Flask or falls back automatically to Python's built-in `ThreadingHTTPServer` if dependencies are missing.
- **Live Performance Stats:** Reports tokens generated, latency in seconds, and generation throughput (tokens/sec).
- **Preset Prompts & Confidence Cards:** Interactive one-click test cases for quick demonstrations.

To launch the web app:
```bash
cd web_app
python app.py
```
Navigate to `http://localhost:5000` in your web browser.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- NVIDIA GPU recommended for training and fast inference (CPU and Apple MPS are fully supported)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/iSHAANphaye/GPT2-From-Scratch.git
   cd GPT2-From-Scratch
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision tiktoken numpy matplotlib jupyterlab flask
   ```

### Running the Notebooks
To explore the implementation step-by-step:
```bash
jupyter lab
```
- Open `GPT2-Single-Notebook/GPT2_From_Scratch.ipynb` for the consolidated all-in-one pipeline.
- Or explore directories `1` through `4` for modular deep-dives into each individual component.

---

## 📊 Results & Benchmarks

| Task / Model | Base Model | Objective | Final Metric |
| :--- | :---: | :---: | :---: |
| **Spam Classification** | GPT-2 Small (124M) | Cross-Entropy Loss | **95.67% Test Accuracy** (97.32% Val) |
| **Instruction Tuning (SFT)** | GPT-2 Medium (355M) | Next-Token Prediction | Stable multi-turn task completion |
| **Pretraining Demo** | GPT-2 Small (from scratch) | Next-Token Prediction | Decreasing loss curve on sample corpus |
| **Inference Engine** | GPT-2 (124M / 355M) | Top-$k$ + Temperature | Sub-second latency on modern GPUs |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
