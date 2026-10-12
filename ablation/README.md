# Ablation — Hindi GPT with positional embeddings removed

Retrain the Hindi model with the learned absolute positional table removed,
run the full Phase 2 evaluation suite on it, and compare it to the standard
Hindi model.

Two Kaggle kernels, run one after the other:

| | what it does | time |
| --- | --- | ---: |
| 1. train | one epoch, 16,125 steps, then pushes the model to Hugging Face | ~5.1 h |
| 2. eval | pulls both models from the Hub, runs the full suite, writes the comparison | ~0.5 h |

Total ≈ 5.6 h. Training checkpoints land every 500 steps, so a timeout costs
minutes rather than the run.

## What "ablated" means here

`positional: "none"` removes the 512 x 384 table entirely — 196,608 fewer
parameters — rather than zeroing it, so it holds no parameters and receives no
gradient. Everything else is held constant against the standard model: same
16K tokenizer, same corpus and split, same one-epoch horizon, same recipe
(peak LR 3e-3, 750 warmup, 32 x 2 x 512 tokens per step), same seed. The only
variable is the table.

A caution worth stating in the write-up: this does **not** make the model
order-blind. Causal attention is itself a positional signal, because position
*t* attends to 0..*t*. A one-layer no-positional model is genuinely
permutation-invariant over its prefix; from two layers on it is not. The
ablated model has to infer order from the mask instead of reading it from an
embedding. Kernel 2 measures that directly with an order-sensitivity probe.

---

## Step 1 — upload the code

Build the archive:

```bash
cd /home/prateek/LMA_project1
bash ablation/make_zip.sh       # -> artifacts/lma-ablation-code.zip
```

On Kaggle: **Datasets → New Dataset**, upload the zip. The original run used the
dataset `prateek99491/lma-bonus-code` (published name kept), built from the
course-era layout; the kernels under `kernels/` are that historical record.

No corpus upload is needed — the token bins are already on Kaggle as the
output of the `lma-phase-2-export-bins-16k` kernel.

## Step 2 — add the Hugging Face token

Create a **write-scoped** token at huggingface.co/settings/tokens.

In the notebook: **Add-ons → Secrets → Add secret**, label `HF_TOKEN`, paste
the value, and attach it to the notebook. It is never printed and never
written to a file. Revoke it after the run.

## Step 3 — run the training kernel

New notebook, paste `kernels/train/ablation_train_nopos.py`, then set:

- **Accelerator**: GPU T4
- **Internet**: on
- **Add Data**: the ablation code dataset (`prateek99491/lma-bonus-code` for the original run)
- **Add Data → Notebook Output**: `lma-phase-2-export-bins-16k`

Then **Save Version → Save & Run All (Commit)**.

It prints the step count and the estimated hours before training starts. At
the end it packages the weights as safetensors and pushes them to
`Prateek-Tiwari10/LMA_bonus_nopos_hindi` (published name kept).

**If it times out**, re-push the same notebook with its own previous output
attached as an extra input. `restore_previous_output()` copies the checkpoints
back and training resumes with the optimiser moments and schedule position
intact. Without that step it would silently restart from scratch.

## Step 4 — run the evaluation kernel

New notebook, paste `kernels/eval/ablation_eval_nopos.py`, same settings, same
two inputs. **Save & Run All (Commit)**.

It fetches both models from the Hub, checks each one's published config before
using it — a run that evaluated the standard model twice would produce a
perfectly formatted comparison showing no difference — and then runs, for each:

- perplexity, bits per byte, bits per token (test split)
- BLEU-4, chrF2, ROUGE-L at greedy and temperature 0.5 / 1.0 / 1.5
- repetition rate and Distinct-1/2 at each of those settings
- attention heatmaps, per-layer entropy and mean attention distance
- the order-sensitivity probe

Outputs land in `/kaggle/working/ablation_eval/`:

- `comparison.md` — the side-by-side table with deltas
- `comparison.json` — every raw number
- `eval_*.json`, `generation_*.json`, `attention_*.json`, `order_*.json`

## Step 5 — bring the results home

Download the eval kernel's output and copy `comparison.md` plus the JSONs into
`reports/ablation/`, then revoke the HF token. Locally the same suite runs with
`bash ablation/run_eval.sh`.

---

## Files

```
ablation/
├── configs/
│   ├── model_16k_nopos.json    positional: "none", 25,663,872 params
│   └── train_16k_nopos.json    identical recipe to the standard run
├── kernels/
│   ├── train/ablation_train_nopos.py   kernel 1
│   └── eval/ablation_eval_nopos.py     kernel 2
├── push_hf.py                  packages and uploads, reads HF_TOKEN from Secrets
├── from_hub.py                 rebuilds a loadable checkpoint from published files
├── order_probe.py              order-sensitivity probe
├── position_decoding.py        linear probe: where position is rebuilt
├── run_eval.sh                 local evaluation suite, standard vs ablated
├── make_zip.sh                 builds artifacts/lma-ablation-code.zip
└── dataset-metadata.json
```

The ablation itself is three small changes in the shared model code, not a
separate fork: `model/config.py` registers the `16k_nopos` variant and accepts
`positional: "none"`, `model/gpt.py` skips building and adding the table, and
`tests/test_model_checks.py` stops assuming it exists.

Checkpoint (gitignored): `artifacts/checkpoints/ablation/ablated_hindi.pt`;
results: `reports/ablation/`.
