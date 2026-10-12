# RESULTS — Stage 1b/1c: Hindi-only restructure and verification

**Status:** done. Both C3 checks and the C4 check reproduced their recorded numbers *before* any old Hindi or shared folder was deleted. **No git command was run**, and `.git` was not touched.

---

## 1. Verification (C3, C4): new layout vs recorded values

All runs used the local RTX 3050 4 GB, torch 2.13.0+cu130, fp16 autocast as in the original runs.

| Check | Measured (new layout) | Recorded | Abs. diff | Result |
|---|---|---|---|---|
| GPT pretrained `final_16k.pt`: test CE, full split (35,903,488 tokens) | 3.4079521429 | 3.4079521467 | 3.8e-9 | **PASS** |
| … perplexity | 30.203329 | 30.203329 | 1.1e-7 | **PASS** |
| … bits per byte (360,869,390 bytes) | 0.48916415 | 0.48916415 | 5.4e-10 | **PASS** |
| GPT-ft `ft_60k/best.pt`: held-out-name val EM (fixed 500-example subset) | **0.954 (477/500)** | 0.954 | 0 | **PASS** |
| … training-split tokenisation (confirms `train_60k.jsonl` is the file it trained on) | 60,000 ex / 5,782,373 tok / 3,523,166 supervised | identical | 0 | **PASS** |
| Ablation `ablated_hindi.pt` through the new shared model code: test CE, 500 × 8 × 512 prefix | 3.3949284329 | 3.3949284329 | 0.0 | **PASS** |
| GPT on the same prefix (cross-check against `eval_standard.json`) | 3.3706493731 | 3.3706493731 | 0.0 | **PASS** |
| New `model/gpt.py` vs old `lm/model.py`: logits and attention on the real checkpoint | max abs diff **0.0** | — | — | bit-identical |
| `tests/test_model_checks.py` (future-blindness, shapes, causal mask, parameter count 25,860,480, initial loss), run before and after deletion | all PASS | — | — | **PASS** |

The EM and model checks were re-run **after** the old folders were deleted, giving the same results. Nothing still depends on `lm/` or `phase3/`.

Reports: `reports/verify_gpt_baselines.json` and `reports/ablation/verify_ablation_reload.json`. Re-run with `python -m tests.verify_gpt_baselines`.

---

## 2. What was deleted

| Group | Paths | Size |
|---|---|---:|
| **Non-Hindi course material, approved plan** | 153 paths: the second-language corpus, tokenizers, reasoning data, checkpoints, kernels, logs, figures and staging copies | **20 GB** |
| Redundant sibling folder | `/home/prateek/hindi_arch_study/` (it only held a copy of the Stage 0 report) | 12 KB |
| **Old Hindi/shared code and staging, C3** (deleted only after §1 passed) | `lm/`, `phase3/` (incl. the `scaling_hindi_{10,30,50}k` checkpoints and `report_pdf.py`), `common/`, `bonus/`, `kaggle/phase2/`, `kaggle/phase3*/`, `kaggle/*.ipynb`, `test/`, `.kaggle_code/`, `.kaggle_stage*/`, `.kaggle_output/`, `extra/`, `converted_jsonl/`, `sweep_outputs/`, `tokenizers/`, `hindi/` (incl. the 24K model `final.pt`), `logs/` | 10.9 GB |
| **Course write-ups, C2** | `Report.pdf`, `report/final/`, `report/final.zip`, `report/phase1/`, `report/phase2/`, `report/phase3/`, `report/bonus/`, `report/fonts/`, `report/scaling_curves.png` (two-language comparison), phase write-ups `phase3/RESULTS_*.md` | ~45 MB |

**Kept untouched:** `report/README.md`, `report/RESULTS_scaling.md` and `report/PROJECT_SUMMARY_FOR_RESUME.md`, your local notes. `/report/` is now gitignored.

**Safety copies, not part of the repository:**
- **Course write-ups:** `phase3/RESULTS_phase3.md` had an **uncommitted local edit** that your backup might not contain, so every write-up and PDF was copied to `artifacts/archive/course_writeups/` (7.8 MB, gitignored) before deletion.
- **Unlisted Hindi results:** anything Hindi that wasn't on an approved deletion list was **moved, not deleted**:
  - the raw Hindi corpus → `artifacts/corpus/` (17 GB);
  - the 24K bins → `artifacts/data/bin_24k_unused/`;
  - the 24K and 4,500-step results, and the 10k/30k/50k raw eval rows → `artifacts/archive/`;
  - Phase 1 / Phase 2 logs → `artifacts/logs/`.

---

## 3. What moved where (old → new)

### Code

| Old | New |
|---|---|
| `lm/config.py` | `model/config.py` (Hindi-only `Paths` pointing at the new layout; YAML or JSON configs) |
| `lm/model.py` | `model/gpt.py` (behaviour bit-identical; causal mask and parameter count come from `model/common.py`) |
| `lm/generate.py::generate` | `model/common.py` (shared sampler) |
| — | `model/__init__.py::build_model(config, device, arch)`: one factory for every architecture |
| `lm/train.py` | `train/pretrain.py` (`--arch`; arch recorded in each checkpoint; refuses a cross-arch resume) |
| `lm/sweep.py` | `train/lr_sweep.py` (default grid = GPT's {1.5e-3, 3e-3, 4.5e-3, 6e-3}; `--arch`); plotting → `viz/lr_sweep_plots.py` |
| `lm/hub.py` | `train/hub.py` |
| `phase3/sft/train.py`, `config.py`, `sweep.py` | `train/finetune.py` (`--arch`, `--base`, explicit split files), `train/finetune_config.py`, `train/finetune_sweep.py` |
| `lm/evaluate.py` | `eval/lm_metrics.py` (architecture-aware loader) |
| `lm/generate.py` (metrics) | `eval/generation_metrics.py` |
| `lm/attention.py` | `eval/attention.py` (computation) + `viz/attention_plots.py` (plots) |
| `lm/compare.py` | `eval/compare.py` |
| `phase3/eval/matrix.py` (+ `code_of` from `run.py`) | `eval/reasoning_eval.py` (tags `pretrained`, `ft_60k`; `--checkpoint` for new models) |
| `phase3/eval/{metrics,parser,analysis}.py` | `eval/reasoning_{metrics,parser,analysis}.py` |
| `phase3/eval/{perplexity,generation_metrics,forgetting}.py` | `eval/reasoning_perplexity.py`, `eval/reasoning_generation.py`, `eval/forgetting.py` (`--model TAG=PATH` for extra checkpoints) |
| `phase3/eval/attention.py` | `eval/attention_reasoning.py` + `viz/attention_plots.py` |
| `phase3/sft/generate.py` | `eval/completions.py` |
| `lm/curves.py`, `phase3/sft/curves.py` | `viz/pretrain_curves.py`, `viz/finetune_curves.py` (keyed by arch, not language) |
| `lm/data.py`, `lm/overlap.py`, `phase3/sft/data.py` | `data/token_stream.py`, `data/overlap.py`, `data/sft_data.py` |
| `phase3/data_gen/*` (generator, run, checks, expand_names, balance_pools, make_subsets, build_scaling_tree, `hi*`/`build*` configs, `hi_*` name pools, manifest) | `data/reasoning/` |
| `common/*.py` (18 files), `common/models/lid_hi_ne.json`, `common/models/cache/wikipedia_hi_4000.json`, `hindi/configs/sources.json` | `data/pipeline/` (reference code; see §4) |
| `lm/verify.py`, `lm/gates.py`, `phase3/sft/gates.py` | `tests/test_model_checks.py` (future-blindness and more), `tests/test_pretrain_gates.py` (overfit one batch, kill-and-resume), `tests/test_finetune_gates.py` (P1–P7 incl. kill-and-resume); plus new `tests/verify_gpt_baselines.py` |
| `bonus/*` | `ablation/` (see §5) |
| `tokenizers/hindi/tokenizer/*` (byte-identical to `hindi/tokenizer/*`) | `tokenizer/` |

**Not ported:** these generated the course write-ups or served superseded runs, so they were dropped:
- `phase3/eval/{report,report_perf,report_scaling,figures,figures_perf,canonical,gates,run}.py`
- `phase3/sft/{baseline,samples,report,push_checkpoints,push_scaling}.py`
- `phase3/data_gen/{report,report_scaling}.py`
- `phase3/report_pdf.py`, `report/final/make_figures.py`
- the course-era Kaggle kernels.

### Configs (values copied from the checkpoints, verified equal)

| New | Source | Diff vs checkpoint config |
|---|---|---|
| `configs/gpt_25m.yaml` | `final_16k.pt["model_config"]` | none (`==` holds) |
| `configs/pretrain_shared.yaml` | `final_16k.pt["train_config"]` + GPT sweep protocol | `run_name` label only |
| `configs/finetune_shared.yaml` | `ft_60k/best.pt["sft_config"]` | adds explicit split paths (`train_file`, `val_file`, `em_file`) only |

`hindi/configs/train_16k.json` held a stale 4,500-step / 600-warmup recipe from the matched-budget comparison run. It is **not** what produced `final_16k.pt`, which used 16,125 steps and 750 warmup, so it was not carried over.

### Artifacts (gitignored, under `artifacts/`)

| Old | New |
|---|---|
| `hindi/data/bin/*_16k.{bin,json}` | `artifacts/data/bin/` (sha256 re-verified: train/val/test OK) |
| `hindi/model/final_16k.pt` | `artifacts/checkpoints/gpt/final_16k.pt` |
| `phase3/runs/scaling_hindi_60k/` | `artifacts/checkpoints/gpt_ft/ft_60k/` |
| `bonus/checkpoints/ablated_hindi.pt`, `bonus/results/train_16k_nopos_hindi/` | `artifacts/checkpoints/ablation/`, `artifacts/runs/ablation_nopos/` |
| `phase3/data_perf/{train,validation,test}/hi` | `artifacts/finetune_data/` |
| `.hf_bonus/` | `artifacts/hf_cache_ablation/` |

### Results → `reports/` (generated results only)

| Folder | Contents |
|---|---|
| `reports/data/` | Hindi cleaning, collection, export, LID and tokenizer JSON; `bin_checksums_16k.txt` (non-Hindi lines removed, paths updated); Phase 1 figures |
| `reports/gpt_pretrain/` | `train_log.jsonl` (65 val points), `train_summary.json`, LR sweep JSON (rounds 1, 2, all), `eval_16k.json`, `generation_16k.json`, `attention_16k.json`, gates, matched-step compare; 16K figures |
| `reports/gpt_finetune/` | `train_log.jsonl`, `train_summary.json`, `finetune_config.json` (ft_60k); `eval_raw/hi_ft_60k_*.jsonl`; perplexity, forgetting, generation, attention JSON; SFT LR sweeps; HF push evidence; generator summaries; Hindi-only figures |
| `reports/ablation/` | eval / generation / attention / order-probe / position-decoding JSON, training log, figures, reload check |
| `reports/RESULTS_*.md` | stage reports (the former `results/` folder; Stage 0 and 1a moved here) |

**Dropped as two-language comparison figures:**
- `report/phase3/figures/{eval_*,perf_*,sft_loss_curves,sft_val_em,sft_lr_gradnorm,sft_relonly_*}.png`
- `report/scaling_curves.png`

**JSON stripping (approved plan):** non-Hindi keys and entries were removed from 22 JSON files, plus the second-language row of the markdown table inside `eval_16k.json`. A script asserted that every remaining leaf equals its original value. Hindi values and the Hindi sha256 tables are untouched; the `train_60k` hash still reads `46f1c3b6…a668632`.

---

## 4. C1: data pipeline

- **Location:** `data/pipeline/`, 18 files: crawl, discover, fetch, frontier, public_corpus, normalize, lid, train_lid, quality, clean, dedup, audit, export, tokenizer, eval_tokenizer, stats_report, hf_tokenizer and `__init__`.
- **Marking:** every module carries a *"REFERENCE CODE — the token files were built once and are not regenerated"* header.
- **Non-Hindi branches removed:** second-language sources, language choices, cleaning branches and model-card text are gone. No file was deleted wholesale.
- **Kept on purpose:**
  - The Hindi-vs-Nepali language-ID filter (`lid.py`, `train_lid.py`, `models/lid_hi_ne.json`) is labelled *"filter used to keep only Hindi documents"*.
  - `NormalizerConfig.for_nepali()` remains because `train_lid.py` needs it for the classifier's negative examples; it is labelled as used only there.
  - All other comments and docstrings were rewritten Hindi-only (follow-up pass).
- **`py_compile`:** **18/18 files pass**, and pyflakes reports no undefined names.
- **Documentation:** `data/README.md` has one section per step, corpus statistics, the reference-code note, sha256 values and storage locations.

## 5. C4: `bonus/` → `ablation/`

- Code, configs, kernels, README, `run_eval.sh` and `make_zip.sh` were moved and "bonus" renamed to "ablation" throughout. `order_probe.py` and `position_decoding.py` import `model.config` / `model.build_model` / `eval.lm_metrics`.
- `run_eval.sh` and `make_zip.sh` were rewritten for the new layout.
- **Published IDs keep "bonus"** because renaming them would break existing links: HF `Prateek-Tiwari10/LMA_bonus_nopos_hindi`, Kaggle `prateek99491/lma-bonus-code`, `lma-bonus-nopos-{train,eval}`. Each is commented as such.
- The two Kaggle kernel scripts are marked *HISTORICAL RECORD*: their imports are updated, but their input-staging logic describes the course-era zip layout.
- The checkpoint loads through the shared code and reproduces its recorded test CE exactly (§1).

## 6. Other changes

- **`.gitignore`** was rewritten. It ignores `artifacts/`, all weights and bins, `*.jsonl` except `reports/**/*.jsonl`, Kaggle/HF caches, `*.zip`, `*.log` and `/report/`.
- **README:** **no GitHub Classroom references** were found in the README, docs or code. The top-level `README.md` was rewritten with these sections: Data & Tokenizer · GPT Pretraining · LM Evaluation · Generation Quality · Attention Analysis · Reasoning Finetuning · Ablation · GPT vs LLaMA (placeholder) · layout · reproduction · HF/Kaggle links.
- **New small READMEs:** `tokenizer/README.md`, `kaggle/README.md` and `tests/README.md`.
- **Smoke tests:** all 32 module entry points import and pass `--help` after the deletion. pyflakes is clean apart from pre-existing unused imports/variables.
- **pyflakes install:** `pyflakes` was pip-installed into `.venv` to lint the port. It is a dev tool and is not added to `requirements.txt`.

## 7. Open items / things to know

1. **`requirements.txt` has no `torch` line**, and it didn't before either. Local is `torch==2.13.0+cu130`. Should I pin it?
2. **24K bins** (`artifacts/data/bin_24k_unused/`, 1.1 GB) and the **raw corpus** (`artifacts/corpus/`, 17 GB) are kept because they weren't on a deletion list. Both are gitignored; delete them if you don't need them.
3. **Drive link for the token bins** is still to be filled in by you (`data/README.md` §4, `README.md`).
4. **The HF repos still contain the course's second-language folders.** The README links only to the Hindi subpaths; I did not touch anything on HF.
5. **`model/llama.py` and `configs/llama_25m.yaml` don't exist yet.** Stage 2 adds them. `ARCHITECTURES` is currently `("gpt",)`, and `--arch` is already wired through pretrain, finetune, sweeps and eval.
6. **No LLaMA training will start without a ping to you first**, as you asked. Stage 2's gates are CPU-only, and Stage 3 needs your go.

---

## 8. Final tree (2 levels; `artifacts/` contents abbreviated)

```
.gitignore  README.md  requirements.txt
ablation/   README.md configs/ kernels/{train,eval}/ order_probe.py position_decoding.py from_hub.py push_hf.py run_eval.sh make_zip.sh dataset-metadata.json
artifacts/  (gitignored) archive/ checkpoints/{gpt,gpt_ft/ft_60k,ablation} corpus/ data/{bin,bin_24k_unused} finetune_data/ hf_cache_ablation/ logs/ runs/ verify/
configs/    gpt_25m.yaml pretrain_shared.yaml finetune_shared.yaml
data/       README.md token_stream.py sft_data.py overlap.py pipeline/ (18 .py + models/ + sources_hindi.json) reasoning/ (generator, configs/, vocab/, manifest_scaling.json)
eval/       lm_metrics.py generation_metrics.py attention.py attention_reasoning.py compare.py completions.py forgetting.py reasoning_{eval,metrics,parser,analysis,perplexity,generation}.py
kaggle/     README.md
model/      __init__.py (build_model) common.py config.py gpt.py
report/     (gitignored local notes) README.md RESULTS_scaling.md PROJECT_SUMMARY_FOR_RESUME.md
reports/    RESULTS_stage0_inventory.md RESULTS_stage1b_cleanup.md verify_gpt_baselines.json data/ gpt_pretrain/ gpt_finetune/ ablation/
tests/      README.md test_model_checks.py test_pretrain_gates.py test_finetune_gates.py verify_gpt_baselines.py
tokenizer/  README.md hindi_bpe_16000.{model,vocab} hindi_bpe_24000.{model,vocab}
train/      pretrain.py lr_sweep.py finetune.py finetune_config.py finetune_sweep.py hub.py
viz/        pretrain_curves.py finetune_curves.py attention_plots.py lr_sweep_plots.py
```
