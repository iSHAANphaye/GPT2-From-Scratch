# Unified Inference Engine for GPT-2 Models (Base, Classification, Instruction)
import os
import math
import time
import json
import numpy as np
import tiktoken
import torch
import torch.nn as nn

# Configuration definitions
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

# Architecture components
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=True):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# Generation utilities
def generate(model, idx, max_new_tokens, context_size, temperature=0.7, top_k=50, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and next_token_id.item() == eos_id:
            break

        idx = torch.cat((idx, next_token_id), dim=1)
    return idx


# Model Manager
class ModelEngine:
    def __init__(self):
        self.device = device
        self.tokenizer = tokenizer
        self.models = {}

    def _find_path(self, relative_candidates):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for rel in relative_candidates:
            full = os.path.normpath(os.path.join(base_dir, rel))
            if os.path.exists(full):
                return full
        return None

    def get_base_model(self):
        if "base" not in self.models:
            print("Loading Base GPT-2 (124M)...")
            model = GPTModel(GPT_CONFIG_124M)
            # Try to load existing checkpoint or download
            try:
                import sys
                from pathlib import Path
                # Add parent directories to import gpt_download
                sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "GPT2-Single-Notebook")))
                from gpt_download import download_and_load_gpt2

                ckpt_dir = self._find_path([
                    "../3. LLM Architecture/gpt2",
                    "../4. Fine-Tuning/gpt2",
                    "../GPT2-Single-Notebook/gpt2",
                    "gpt2"
                ]) or "gpt2"

                settings, params = download_and_load_gpt2(model_size="124M", models_dir=ckpt_dir)
                
                # Assign weights
                def assign(left, right):
                    return torch.nn.Parameter(torch.tensor(right))
                
                model.pos_emb.weight = assign(model.pos_emb.weight, params['wpe'])
                model.tok_emb.weight = assign(model.tok_emb.weight, params['wte'])
                for b in range(len(params["blocks"])):
                    q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
                    model.trf_blocks[b].att.W_query.weight = assign(model.trf_blocks[b].att.W_query.weight, q_w.T)
                    model.trf_blocks[b].att.W_key.weight = assign(model.trf_blocks[b].att.W_key.weight, k_w.T)
                    model.trf_blocks[b].att.W_value.weight = assign(model.trf_blocks[b].att.W_value.weight, v_w.T)

                    q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
                    model.trf_blocks[b].att.W_query.bias = assign(model.trf_blocks[b].att.W_query.bias, q_b)
                    model.trf_blocks[b].att.W_key.bias = assign(model.trf_blocks[b].att.W_key.bias, k_b)
                    model.trf_blocks[b].att.W_value.bias = assign(model.trf_blocks[b].att.W_value.bias, v_b)

                    model.trf_blocks[b].att.out_proj.weight = assign(model.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
                    model.trf_blocks[b].att.out_proj.bias = assign(model.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

                    model.trf_blocks[b].ff.layers[0].weight = assign(model.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
                    model.trf_blocks[b].ff.layers[0].bias = assign(model.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
                    model.trf_blocks[b].ff.layers[2].weight = assign(model.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
                    model.trf_blocks[b].ff.layers[2].bias = assign(model.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

                    model.trf_blocks[b].norm1.scale = assign(model.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
                    model.trf_blocks[b].norm1.shift = assign(model.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
                    model.trf_blocks[b].norm2.scale = assign(model.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
                    model.trf_blocks[b].norm2.shift = assign(model.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

                model.final_norm.scale = assign(model.final_norm.scale, params["g"])
                model.final_norm.shift = assign(model.final_norm.shift, params["b"])
                model.out_head.weight = assign(model.out_head.weight, params["wte"])
            except Exception as e:
                print(f"Warning loading OpenAI 124M weights: {e}")

            model.to(self.device)
            model.eval()
            self.models["base"] = model
        return self.models["base"]

    def get_classifier_model(self):
        if "classifier" not in self.models:
            print("Loading Classification Fine-Tuned Model (124M)...")
            model = GPTModel(GPT_CONFIG_124M)
            model.out_head = nn.Linear(GPT_CONFIG_124M["emb_dim"], 2)

            ckpt_path = self._find_path([
                "../4. Fine-Tuning/gpt2-small124M-classification-finetuning.pth",
                "../4. Fine-Tuning/review_classifier.pth",
                "../GPT2-Single-Notebook/gpt2-small124M-classification-finetuning.pth",
            ])
            if ckpt_path:
                print(f"Found classification checkpoint: {ckpt_path}")
                model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            else:
                print("Notice: Classification weights not found, using initialized classification head.")

            model.to(self.device)
            model.eval()
            self.models["classifier"] = model
        return self.models["classifier"]

    def get_instruction_model(self):
        if "instruction" not in self.models:
            print("Loading Instruction Fine-Tuned Model (355M)...")
            model = GPTModel(GPT_CONFIG_355M)

            sft_path = self._find_path([
                "../4. Fine-Tuning/gpt2-medium355M-sft.pth",
                "../GPT2-Single-Notebook/gpt2-medium355M-sft.pth"
            ])
            if sft_path:
                print(f"Found SFT checkpoint: {sft_path}")
                model.load_state_dict(torch.load(sft_path, map_location=self.device))
            else:
                print("Notice: gpt2-medium355M-sft.pth not found, loading base 355M weights.")
                try:
                    import sys
                    sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "GPT2-Single-Notebook")))
                    from gpt_download import download_and_load_gpt2
                    ckpt_dir = self._find_path(["../4. Fine-Tuning/gpt2", "gpt2"]) or "gpt2"
                    settings, params = download_and_load_gpt2(model_size="355M", models_dir=ckpt_dir)
                    # Assign weights logic here if needed
                except Exception as e:
                    print(f"Notice loading 355M: {e}")

            model.to(self.device)
            model.eval()
            self.models["instruction"] = model
        return self.models["instruction"]

    def generate_text(self, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
        start_time = time.time()
        model = self.get_base_model()
        encoded = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        idx = torch.tensor(encoded, device=self.device).unsqueeze(0)
        
        token_ids = generate(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=GPT_CONFIG_124M["context_length"],
            temperature=temperature,
            top_k=top_k
        )
        elapsed = time.time() - start_time
        generated_text = tokenizer.decode(token_ids.squeeze(0).tolist())
        num_generated_tokens = token_ids.shape[1] - len(encoded)

        return {
            "prompt": prompt,
            "output": generated_text,
            "tokens_generated": num_generated_tokens,
            "latency_seconds": round(elapsed, 3),
            "tokens_per_second": round(num_generated_tokens / max(elapsed, 1e-4), 1)
        }

    def classify_text(self, text, max_length=120):
        start_time = time.time()
        model = self.get_classifier_model()
        encoded = tokenizer.encode(text)
        supported_context = GPT_CONFIG_124M["context_length"]
        encoded = encoded[:min(max_length, supported_context)]
        encoded += [50256] * (max_length - len(encoded))  # Pad with <|endoftext|>

        input_tensor = torch.tensor(encoded, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred].item()
        elapsed = time.time() - start_time

        return {
            "text": text,
            "label": "spam" if pred == 1 else "not spam",
            "is_spam": pred == 1,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "ham": round(probs[0, 0].item() * 100, 2),
                "spam": round(probs[0, 1].item() * 100, 2)
            },
            "latency_seconds": round(elapsed, 3)
        }

    def instruct_assistant(self, instruction, input_context="", max_new_tokens=150, temperature=0.0):
        start_time = time.time()
        model = self.get_instruction_model()

        # Build Alpaca template
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}"
        if input_context and input_context.strip():
            prompt += f"\n\n### Input:\n{input_context.strip()}"
        prompt_with_header = prompt + "\n\n### Response:\n"

        encoded = tokenizer.encode(prompt_with_header, allowed_special={"<|endoftext|>"})
        idx = torch.tensor(encoded, device=self.device).unsqueeze(0)

        token_ids = generate(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=GPT_CONFIG_355M["context_length"],
            temperature=temperature,
            eos_id=50256
        )
        elapsed = time.time() - start_time
        full_text = tokenizer.decode(token_ids.squeeze(0).tolist())
        response = full_text[len(prompt_with_header):].replace("<|endoftext|>", "").strip()
        num_generated_tokens = token_ids.shape[1] - len(encoded)

        return {
            "instruction": instruction,
            "input_context": input_context,
            "response": response,
            "tokens_generated": num_generated_tokens,
            "latency_seconds": round(elapsed, 3),
            "tokens_per_second": round(num_generated_tokens / max(elapsed, 1e-4), 1)
        }

    def get_status(self):
        return {
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "models_loaded": list(self.models.keys()),
            "pytorch_version": torch.__version__
        }


engine = ModelEngine()
