# GPT-Style-Transformer (Personal Learning Project)
A PyTorch implementation of a small transformer trained on English Wikipedia for next-token prediction. Built as a learning exercise.

---

## Architecture

| Component | Detail |
|---|---|
| Embedding dim (`DIM`) | 64 |
| Vocabulary | GPT-2 BPE — 50,257 tokens (`tiktoken`) |
| Context length (`block_size`) | 64 tokens |
| Attention | Single `nn.MultiheadAttention` layer, 4 heads |
| Normalization | Pre-residual `LayerNorm` |
| Output | Linear projection → logit over vocabulary |
| Parameters (approx.) | ~6.7M (dominated by embedding tables) |



---

## Data

- **Source:** `wikimedia/wikipedia` — `20231101.en` split (via HuggingFace `datasets`)
- **Subset used:** first 100,000 articles concatenated into a flat token sequence
- **Tokenizer:** GPT-2 BPE (`tiktoken`)
- **Windowing:** sliding window over the token sequence, `block_size=64`, `step=32` (50% overlap between consecutive windows), producing `(x, y)` pairs where `y` is `x` shifted by one token



---

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Batch size | 32 |
| Epochs | 1 |
| Precision | Mixed (FP16 via `torch.amp.autocast`) |
| Compilation | `torch.compile` |
| Device | CUDA (falls back to CPU) |

Loss is tracked per batch. No learning rate schedule or gradient clipping is applied.

---

## Inference

Generation uses autoregressive sampling with temperature scaling:

```python
generate(model, prompt="The history of", max_new_tokens=200, temperature=0.8)
```

The context is cropped to `block_size` tokens when the generated sequence exceeds it. Sampling is done via `torch.multinomial` over the softmax distribution.

---

## Requirements
Install:

```bash
pip install -r requirements.txt
```

Originally developed on Google Colab with a T4 GPU (`accelerator: GPU`).

---

## Known Issues and Limitations
- **Definition order:** `VOCAB_SIZE` and `DIM` are defined *after* the `LanguageModel` class. The class body references these as globals, so instantiation will fail if the cells are run in notebook order. Move the constant definitions before the class.
- **No validation split:** There is no held-out set, so overfitting cannot be detected.
- **Scale:** At `DIM=64` with a single attention layer and one epoch of training, the model's outputs will be syntactically plausible at best. Do not expect coherent long-form text.
- **EOT token defined but unused:** `EOT_TOKEN` is extracted from the encoder but never inserted between documents during encoding, meaning document boundaries are invisible to the model.
---

## File Structure

```
llm_training.ipynb   # All training, data processing, and generation code
llm.pth              # Saved model weights (produced after training)
```

---

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI. (GPT-2)
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI BPE tokenizer
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) — Wikipedia corpus
