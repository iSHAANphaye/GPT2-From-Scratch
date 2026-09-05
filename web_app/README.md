# GPT-2 Interactive Studio (Web Interface)

A web interface allowing you to interact with all three models developed in this project:
1. **AI Assistant**: Instruction fine-tuned model (Alpaca SFT 355M)
2. **Spam Detector**: Classification fine-tuned model (SMS Spam 124M)
3. **Creative Generator**: Base pretrained generative model (GPT-2 124M)

---

## Quick Start

### 1. Launch the Server
From this directory, simply run:
```bash
python app.py
```
*(No external dependencies like Flask are strictly required; if Flask is not installed, the app automatically runs on Python's built-in `http.server`!)*

### 2. Open in Your Browser
Open your browser and navigate to:
```
http://localhost:5000
```

---

## Features

- **Tab 1: AI Assistant (Instruction Model)**
  - Enter any instruction (e.g. grammar editing, similes, Q&A, unit conversions).
  - Quick-click preset chips for instant demonstrations.
  - Automatically wraps inputs in the Alpaca prompt template.
- **Tab 2: Spam Detector (Classification Model)**
  - Enter or paste any SMS message.
  - Visual verdict card displaying whether the message is **SPAM** or **NOT SPAM (HAM)** along with confidence score.
  - Includes sample spam/ham buttons to test edge cases.
- **Tab 3: Creative Generator (Base Model)**
  - Free-form text completion.
  - Adjustable **Temperature** (creativity) and **Top-K** (sampling diversity) sliders.
  - Displays real-time generation speed (tokens per second) and token count.

---

## Model Weight Resolution
The server automatically scans and reuses existing weights already in your repository:
- `../4. Fine-Tuning/gpt2-medium355M-sft.pth` (Instruction Assistant)
- `../4. Fine-Tuning/gpt2-small124M-classification-finetuning.pth` (Spam Classifier)
- `../3. LLM Architecture/gpt2/124M` (Base Pretrained Model)
No weights are redownloaded from the internet.
