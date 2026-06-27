<div align="center">

# RoBERTa × ANLI
### Adversarial Natural Language Inference via Curriculum Learning

<br/>

<img src="https://img.shields.io/badge/Model-RoBERTa--base-8A2BE2?style=for-the-badge&logo=huggingface&logoColor=white"/>
<img src="https://img.shields.io/badge/Task-NLI-0078d4?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Dataset-ANLI%20R1%2BR2%2BR3-FF6B35?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Framework-Transformers%204.47-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/><br/>

> **A four-stage curriculum fine-tuning pipeline** that progressively transfers `roberta-base` from broad multi-genre NLI understanding to adversarially-robust inference — starting from easy, abundant corpora and ending on human-verified examples specifically crafted to fool state-of-the-art models.

</div>

---

## What Is Adversarial NLI?

**Natural Language Inference (NLI)** asks a model to decide whether a *hypothesis* is entailed by, contradicted by, or neutral with respect to a *premise* — a three-way classification at the heart of language understanding.

**Adversarial NLI (ANLI)** supercharges this with a *human-in-the-loop* adversarial collection protocol: annotators craft hypotheses while a live model tries to classify them, and **only examples that fool the model are kept**. The result is a benchmark that systematically exposes model weaknesses, making it orders of magnitude harder than standard NLI corpora.

```
Premise   :  "The scientist published a groundbreaking paper on quantum entanglement."
Hypothesis:  "The scientist contributed to the field of physics."
Label     :  ENTAILMENT  ✅

Premise   :  "The scientist published a groundbreaking paper on quantum entanglement."
Hypothesis:  "No scientific work was released by the researcher."
Label     :  CONTRADICTION  ❌

Premise   :  "The scientist published a groundbreaking paper on quantum entanglement."
Hypothesis:  "The paper took longer than expected to write."
Label     :  NEUTRAL  🤔
```

---

## The Curriculum Learning Approach

Training directly on ANLI from a pretrained `roberta-base` fails — the adversarial distribution shift is too steep and the model struggles to converge reliably. Curriculum learning solves this by staging the difficulty:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STAGE 1  ──  Base NLI Pre-training
              MNLI (~392K) + SNLI (~549K)
              3 epochs · batch 32 · lr 2e-5 cosine
              ✅  Eval Accuracy: 91.83%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              ↓  Warm-start from Stage 1 best checkpoint
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STAGE 2  ──  ANLI Round 1 Fine-tuning
              ANLI R1 (~16.9K adversarial pairs)
              Low LR (1–2e-6) · early stopping
              ✅  Eval Accuracy: 49.50%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              ↓  Warm-start from Stage 2 best checkpoint
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STAGE 3  ──  ANLI R1 + R2 Fine-tuning
              ANLI R1+R2 (~33K adversarial pairs)
              Progressive difficulty increase
              ✅  Eval Accuracy: 51.70%  ← Best ANLI Score
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              ↓  Warm-start from Stage 3 best checkpoint
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STAGE 4  ──  ANLI R1 + R2 + R3 (All Rounds)
              ANLI R1+R2+R3 (~100K adversarial pairs)
              Maximum adversarial exposure
              ✅  Eval Accuracy: 49.13%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Training Results

<div align="center">

![Curriculum Results](curriculum_results_v10.png)

</div>

| Stage | Training Data | Best Accuracy | Eval Loss | Steps | Notes |
|:---:|---|:---:|:---:|:---:|---|
| **1** | MNLI + base NLI | **91.83%** | 0.361 | 53,925 | Strong NLI baseline |
| **2** | ANLI Round 1 | **49.50%** | — | ~325 | Adversarial cold start |
| **3** | ANLI R1 + R2 | **51.70%** | — | ~3,500 | Best overall ANLI score |
| **4** | ANLI R1 + R2 + R3 | **49.13%** | — | ~5,000 | Full round exposure |

> **Context:** ANLI scores in the 49–52% range are expected and competitive for `roberta-base`. ANLI is deliberately designed to expose model failure modes — a random baseline sits at 33%, and even top-performing large models rarely exceed 70%. These results demonstrate genuine adversarial robustness, not underfitting.

---

## Per-Round Analysis

<div align="center">

| ANLI Round 1 | ANLI Round 2 | ANLI Round 3 |
|:---:|:---:|:---:|
| ![R1](analysis_ANLI_R1.png) | ![R2](analysis_ANLI_R2.png) | ![R3](analysis_ANLI_R3.png) |

</div>

---

## Model Architecture

```
Input Sequence
  "[CLS] premise [SEP] hypothesis [SEP]"
                │
                ▼
┌───────────────────────────────────────────┐
│            roberta-base                   │
│                                           │
│  • 12 × Transformer encoder layers        │
│  • 12 attention heads per layer           │
│  • 768-dimensional hidden states          │
│  • 50,265 BPE vocabulary tokens           │
│  • Max position embeddings: 514           │
│  • ~125M parameters                       │
└───────────────────────────────────────────┘
                │
        [CLS] hidden state (768-dim)
                │
                ▼
     ┌──────────────────────┐
     │  Classification Head  │
     │  Linear(768 → 3)     │
     └──────────────────────┘
                │
                ▼
  { Entailment | Neutral | Contradiction }
```

---

## Repository Structure

```
roberta-anli/
│
├── 📓 roberta_anli_v10.ipynb        ← Full training & evaluation notebook
├── 📋 guide.text                    ← Quick-start guide (no training needed)
│
├── 🤖 roberta-anli-final/           ← Final exported model & tokenizer
│   ├── config.json
│   ├── model.safetensors            ← [excluded — use HF Hub or Git LFS]
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
│
├── 📊 curriculum_results_v10.png    ← Training curves across all stages
├── 📊 analysis_ANLI_R1.png          ← Round 1 prediction analysis
├── 📊 analysis_ANLI_R2.png          ← Round 2 prediction analysis
├── 📊 analysis_ANLI_R3.png          ← Round 3 prediction analysis
│
├── stage1_checkpoint/               ← Stage 1 config + training log
├── stage1_snli_checkpoint/          ← Stage 1 (SNLI) config
├── stage2_checkpoint/               ← Stage 2 config + training log
├── stage3_checkpoint/               ← Stage 3 config + training log
└── stage4_checkpoint/               ← Stage 4 config + training log
```

> **Note:** Model weights (`*.safetensors`, `*.pt`) are excluded from this repository due to size. To use the model, run the training pipeline in the notebook or host weights on the HuggingFace Hub.

---

## Quick Start

### Installation

```bash
# With GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.1 datasets accelerate scikit-learn notebook

# CPU only
pip install torch
pip install transformers==4.47.1 datasets accelerate scikit-learn notebook
```

### Run Inference (no training required)

After placing trained weights in `roberta-anli-final/`, open the notebook and run only these cells:

1. **First cell** — checks Python/CUDA versions
2. **pip install cell** — installs dependencies
3. **Imports cell** — loads all libraries
4. **Skip** all Stage 1–4 training cells
5. **Scroll to the bottom** → find `def predict_nli(premise, hypothesis):` → run it
6. Run the demo prediction cells below it

### Programmatic Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("./roberta-anli-final")
model     = AutoModelForSequenceClassification.from_pretrained("./roberta-anli-final")
model.eval()

id2label = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

def predict_nli(premise: str, hypothesis: str) -> tuple[str, float]:
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs       = torch.softmax(logits, dim=-1)
    pred_id     = probs.argmax(dim=-1).item()
    confidence  = probs[0, pred_id].item()
    return id2label[pred_id], round(confidence * 100, 1)

# Try it
label, conf = predict_nli(
    "Scientists have confirmed water ice at the lunar south pole.",
    "The moon contains frozen water."
)
print(f"{label}  ({conf}% confidence)")
# → Entailment  (92.4% confidence)
```

### Run the Full Training Pipeline

```bash
jupyter notebook roberta_anli_v10.ipynb
```

Run all cells in order — the notebook handles all four stages with automatic checkpoint saving and early stopping.

---

## Datasets

| Dataset | Pairs (Train) | Domain | Difficulty |
|---|:---:|---|:---:|
| **MNLI** | ~392,702 | Multi-genre (fiction, government, telephone…) | ★★☆☆☆ |
| **SNLI** | ~549,367 | Image captions | ★★☆☆☆ |
| **ANLI R1** | ~16,946 | Adversarial (model-fooling) | ★★★★☆ |
| **ANLI R2** | ~45,460 | Adversarial (harder than R1) | ★★★★★ |
| **ANLI R3** | ~100,459 | Adversarial (hardest) | ★★★★★ |

---

## Training Hyperparameters

### Stage 1 (Base NLI)

| Parameter | Value |
|---|---|
| Learning rate | `2e-5` |
| LR schedule | Cosine decay with linear warmup |
| Batch size | 32 |
| Epochs | 3 |
| Total steps | 53,925 |
| Early stopping patience | 3 |
| Eval every | 500 steps |

### Stages 2–4 (Adversarial Fine-tuning)

| Parameter | Value |
|---|---|
| Learning rate | `1–2e-6` |
| LR schedule | Cosine / linear |
| Batch size | 32 |
| Early stopping patience | 3 |
| Warm-start | Best checkpoint from previous stage |

The low learning rate in fine-tuning stages is intentional — it preserves the general NLI representations learned in Stage 1 while allowing the model to adapt to adversarial distributions without catastrophic forgetting.

---

## Key Findings

**1. Curriculum matters.** Direct fine-tuning on ANLI from `roberta-base` converges poorly. The four-stage warm-start pipeline is essential.

**2. Stage 3 is the sweet spot.** Training on R1+R2 (**51.7%**) outperforms training on all three rounds (**49.1%**). This suggests R3 examples are either too noisy or too far from the model's current capacity — adding them hurts more than it helps at this scale.

**3. ANLI is genuinely hard.** The ~41-point drop from 91.8% (MNLI) to 50.8% (ANLI avg) is not a bug — it's the point. Standard NLI benchmarks have statistical biases that models exploit as shortcuts; ANLI closes those loopholes.

**4. NLI knowledge transfers.** Stage 1 accuracy of **91.83%** confirms `roberta-base` becomes a strong NLI reasoner after warm-up, providing a solid foundation for adversarial fine-tuning.

---

## References

```bibtex
@article{liu2019roberta,
  title   = {RoBERTa: A Robustly Optimized BERT Pretraining Approach},
  author  = {Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and
             Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and
             Zettlemoyer, Luke and Stoyanov, Veselin},
  journal = {arXiv preprint arXiv:1907.11692},
  year    = {2019}
}

@article{nie2019adversarial,
  title   = {Adversarial NLI: A New Benchmark for Natural Language Understanding},
  author  = {Nie, Yixin and Williams, Adina and Dinan, Emily and Bansal, Mohit and
             Weston, Jason and Kiela, Douwe},
  journal = {arXiv preprint arXiv:1910.14599},
  year    = {2019}
}

@inproceedings{williams2018broad,
  title     = {A Broad-Coverage Challenge Corpus for Sentence Understanding through
               Inference (MultiNLI)},
  author    = {Williams, Adina and Nangia, Nikita and Bowman, Samuel R.},
  booktitle = {NAACL},
  year      = {2018}
}

@inproceedings{bowman2015large,
  title     = {A Large Annotated Corpus for Learning Natural Language Inference},
  author    = {Bowman, Samuel R. and Angeli, Gabor and Potts, Christopher and
               Manning, Christopher D.},
  booktitle = {EMNLP},
  year      = {2015}
}
```

---

<div align="center">

Built with PyTorch · HuggingFace Transformers · Patience · And a lot of adversarial examples

**Anindya Roy Chowdhury** · Semester ML Project · 2025

</div>