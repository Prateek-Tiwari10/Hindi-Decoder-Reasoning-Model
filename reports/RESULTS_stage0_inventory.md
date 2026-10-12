# RESULTS — Stage 0: Inventory of the course project (read-only)

**Source inspected:** `/home/prateek/LMA_project1` (branch `bonus`). Nothing in it was modified.
**Method:** I read the code, configs, checkpoints, logs and reports. I computed parameter counts by building the model from code, not by copying them from the docs.
**Location of this file:** `/home/prateek/hindi_arch_study/results/`. This is a sibling of the graded repo, not inside it. Stage 1 fills in the rest of this tree.

---

## 1. Directory tree and language classification

```
LMA_project1/
  lm/                 SHARED   Phase 2 model/train/eval code (model.py, config.py, data.py, train.py,
                               evaluate.py, generate.py, attention.py, sweep.py, gates.py, compare.py, curves.py, hub.py)
  common/             SHARED   Phase 1 corpus pipeline (crawl, clean, dedup, LID, export, tokenizer, hf_tokenizer.py)
  hindi/              HINDI    configs/ (model*.json, train*.json, sources.json), data/bin/ (token bins),
                               model/ (final.pt = 24K baseline, final_16k.pt = 16K/384/11L), tokenizer/
  tokenizers/         SHARED   committed copies: tokenizers/hindi/tokenizer/hindi_bpe_{16000,24000}.model
  phase3/             SHARED   data_gen/ (generator + configs hi*/ne*.yaml + vocab/ name pools), data_perf/
                               (train/validation/test for hi AND ne), sft/ (finetune harness), eval/ (reasoning eval,
                               attention, perplexity, generation metrics), runs/scaling_hindi_{10,30,50,60}k/
  bonus/              HINDI    no-positional-embedding ablation (Hindi only)
  kaggle/             SHARED   kernel sources: phase2/{train_16k_hindi, sweep_16k, ...}, phase3*/
  logs/phase2/runs/   SHARED   16k_hindi/, 24k_hindi/, 16k_reduced_hindi/ (log.jsonl + summary.json)
  report/             SHARED   phase1/2/3 JSON evidence + figures, final/ report; report/*.md gitignored (local notes)
  extra/              SHARED   superseded Phase 3 runs/results, instruction docs
  converted_jsonl/, sweep_outputs/, test/, .kaggle_stage*/, .kaggle_code/, .kaggle_output/, .hf_bonus/   staging/scratch
```

Hindi files the new workspace needs: `lm/*` (code), `hindi/configs/model_16k.json` and `train_16k*.json`, `hindi/data/bin/*_16k.{bin,json}`, `tokenizers/hindi/tokenizer/hindi_bpe_16000.*`, `hindi/model/final_16k.pt`, `phase3/{sft,eval,data_gen}` (Hindi config and vocab only), `phase3/data_perf/{train,validation,test}/hi/`, `phase3/runs/scaling_hindi_60k/`, and `logs/phase2/runs/16k_hindi/`.

---

## 2. GPT-style model code

- **File:** `lm/model.py`, with config in `lm/config.py` (`ModelConfig`, `TrainConfig`, `Paths`).
- **Classes:** `MultiHeadCausalSelfAttention`, `FeedForward`, `TransformerBlock`, `DecoderLM`, plus the `build_model()` factory, which asserts that the live parameter count equals the analytic one.

| Component | Implementation |
|---|---|
| Attention | Hand-written MHA. Q, K, V and out are separate `nn.Linear(d,d)` layers **with bias**. Scale is 1/√d_k. The causal mask is additive −inf before the softmax and is a non-persistent buffer of size `context_len²`. The forward pass can return attention weights `(B,H,T,T)`. No SDPA. |
| Positions | **Learned absolute** `nn.Embedding(context_len=512, d)`, added to token embeddings. A sequence with T > 512 raises `ValueError`. |
| Norm | **Pre-norm** `nn.LayerNorm` (eps 1e-5, weight and bias) before attention and before FFN, plus a **final LayerNorm**. |
| FFN | `Linear(d→d_ff) → GELU (exact, not tanh) → Linear(d_ff→d)`, with biases. **d_ff = 1536 = 4d.** |
| Weight tying | **Tied.** `lm_head.weight is token_embedding.weight`, and `lm_head` has no bias. |
| Dropout | 0.1 on the embedding sum, on attention weights (a copy, so the returned weights are clean), on the attention residual and on the FFN output. |
| Init | GPT-2 style: N(0, 0.02) for Linear and Embedding, zero biases, and `out_proj`/`fc_out` at 0.02/√(2·n_layers). |

---

## 3. Exact config and trainable parameter count (computed from code)

There are two Hindi GPT models in the repo. **The study baseline is the 16K narrow-deep model**, because Phase 3 finetuned from it.

| Model | vocab | d_model | n_layers | n_heads | d_k | d_ff | ctx | params (code) | file |
|---|---|---|---|---|---|---|---|---|---|
| **16K/384/11L (baseline for this study)** | 16,000 | 384 | 11 | 6 | 64 | 1536 | 512 | **25,860,480** | `hindi/model/final_16k.pt` |
| 24K/448/7L (Phase 2 baseline, not used here) | 24,000 | 448 | 7 | 8 | 56 | 1792 | 512 | 27,882,176 | `hindi/model/final.pt` |

`DecoderLM(ModelConfig.load("hindi/configs/model_16k.json")).num_parameters()` returns **25,860,480**. This matches the analytic count. `final_16k.pt` loads into it with `strict=True` and all keys match. **Embeddings are tied**, so the 6,144,000-parameter head is counted once.

Per-component breakdown, summed over 11 blocks:

| Component | Params |
|---|---:|
| token_embedding (tied with lm_head) | 6,144,000 |
| position_embedding (512×384) | 196,608 |
| attn q/k/v/out weights (4 × 11 × 384²) | 6,488,064 |
| attn q/k/v/out biases | 16,896 |
| LayerNorms in blocks (2 per block, weight+bias) | 16,896 |
| FFN fc_in weight + bias | 6,488,064 + 16,896 |
| FFN fc_out weight + bias | 6,488,064 + 4,224 |
| final LayerNorm | 768 |
| **Total** | **25,860,480** |

**Early look at parameter matching for Stage 2 (not a commitment).** A LLaMA model with no position table, no biases, RMSNorm (weight only), tied head and SwiGLU hidden 1024 has 6,144,000 + 11·(589,824 + 768 + 1,179,648) + 384 = **25,617,024** parameters. That is **−0.94 %** against GPT, inside ±1 % but close to the edge. Hidden 1056 gives 26,022,528 (+0.63 %). Hidden 1088, the next multiple of 64, overshoots at +2.2 %. I'll confirm this from code in Stage 2.

---

## 4. Tokenizers

| Tokenizer | Local path | md5 |
|---|---|---|
| **Hindi BPE 16,000 (used)** | `tokenizers/hindi/tokenizer/hindi_bpe_16000.model` (+ `.vocab`); identical copy in `hindi/tokenizer/` | `e50d13a005839869fb9b466ef89d2aa7` |
| Hindi BPE 24,000 | `tokenizers/hindi/tokenizer/hindi_bpe_24000.model` | `d2a45da698cb308abebab03ad7a58719` |

The HF repo `Prateek-Tiwari10/lma_phase1-tokenizer` is the default in `common/hf_tokenizer.py`. Every Phase 2 and Phase 3 HF model directory also carries the `.model`/`.vocab` its weights were trained with. EOS id = 2 (`</s>`).

---

## 5. Hindi pretraining data (16K bins)

The bins are local in `hindi/data/bin/`. Each is uint16 with a JSON sidecar. On Kaggle they came from kernel output `prateek99491/lma-phase-2-export-bins-16k`. Checksums are in `report/phase2/data/bin_checksums_16k.txt`.

| Split | File | Tokens | sha1 (first 64 MB) |
|---|---|---:|---|
| train | `train_16k.bin` | **528,404,893** | `fed26595988c369ba075c7bfc958053f262fd67d` |
| val | `val_16k.bin` | 35,779,979 | `c680f3a7fa6157dab4cb2eb197c7bf5679d7c960` |
| test | `test_16k.bin` | 35,903,951 | `4e2e425dba27df23d9f730a12e51e845285bf21c` |

**Ordering and seed.** There is no pre-shuffled order on disk. `TokenStream.sample()` (in `lm/data.py`) draws **uniformly random window start offsets** over the whole train file using `np.random.default_rng(1337)` (PCG64). Each optimiser step draws 2 micro-batches of 32 offsets. Validation uses a fresh `default_rng(1338)` on every evaluation, so all 65 evaluations score the **same 50 × 32 windows**. The data order is therefore a pure function of (bin file, seed, micro_batch, grad_accum, context_len), and **can be reproduced exactly**. See §10.

---

## 6. Hindi pretraining run that was actually used

Sources: `final_16k.pt` (embedded `train_config`), `logs/phase2/runs/16k_hindi/{log.jsonl,summary.json}`, `kaggle/phase2/train_16k_hindi/`, and `report/phase2/lr_sweeps/lr_sweep_16k_*.json`.

| Setting | Value |
|---|---|
| Steps | **16,125**, computed at runtime as train tokens // 32,768 ("one epoch's worth") |
| Tokens per step | 32 micro × 2 accum × 512 = **32,768** |
| **Total tokens seen** | **528,384,000**. Random offsets, so about 63 % of distinct windows are visited, not a strict epoch |
| Optimiser | AdamW, betas (0.9, 0.95), eps 1e-8, wd 0.1 on ≥2-D params only (biases and norms excluded) |
| Grad clip | 1.0, applied after `scaler.unscale_` |
| LR schedule | Linear warmup over **750** steps (lr = peak·(s+1)/750), then cosine to **floor = 0.1·peak = 3e-4** at step 16,125. The schedule is a pure function of step; there is no LRScheduler object. |
| Peak LR | **3e-3** |
| How the peak LR was chosen | **Swept, with a bracketed interior optimum.** Each run was 1,200 steps (39,321,600 tokens, 7.44 % of budget) with warmup 150 and cosine over 1,200 steps. Grid {1.5e-3, 3e-3, 4.5e-3, 6e-3}; final val loss 4.407 / **4.254** / 4.291 / 4.420. The 3e-3 run reproduced bit-for-bit across sessions. |
| Precision | fp16 autocast + `torch.amp.GradScaler`. 4 steps were skipped for fp16 overflow. |
| Dropout | 0.1 |
| Eval | Every 250 steps plus the final step: 50 val batches × 32 seqs × 512. 65 val points in the log. |
| Checkpoints | Every 500 steps (`last.pt`), and `best.pt` on a new best val loss |
| Seed | 1337 (`torch.manual_seed`, `np.random.seed`, data RNG) |
| Hardware | One Kaggle **T4** (`machine_shape: NvidiaTeslaT4`, single-device code, no DDP), wall time 17,023.6 s (4.73 h). No resume was needed: the log has steps 1 to 16,125 contiguous. |
| Final | val loss **3.4033** (step 16,125). Test set (full, `lm/evaluate.py`): CE 3.4080, **PPL 30.20, BPB 0.4892** |

The log (`log.jsonl`) records train_loss, lr, grad_norm, seconds and tokens every step, and val_loss every 250 steps. **It has no val BPB.**

---

## 7. Pretrained checkpoint used for Phase 3

`hindi/model/final_16k.pt`: step 16,125, val loss 3.4033, 310 MB, resume-capable (model, optimiser, scaler, step, RNG). On HF it is `Prateek-Tiwari10/LMA_phase2` → `16k/hindi/` (`model.safetensors` + `checkpoint.pt`). Every Hindi scaling run's `summary.json` gives `base_checkpoint: .../hindi/model/final_16k.pt, base_step 16125`.

---

## 8. Phase 3 finetuning harness

- **Generator:** `phase3/data_gen/` (`generator.py`, `run.py`, `checks.py` with gates G1–G8, `expand_names.py`, `build_scaling_tree.py`, `make_subsets.py`). The Hindi surface config is `configs/hi_perf.yaml` (the "perf" build, identical to `hi.yaml` except for the name pools). Master seed **20260910**, subset seed **20260912**.
- **Name pools (Hindi):**
  - Old Part 2 build: train **243** (`hi_names_train.txt`), test 93.
  - Expanded "perf" build (Fix Cycle 1): train file **3,698** lines and test file **1,184** lines (`hi_names_*_perf.txt`). The effective pools reported in the manifest after filtering are **3,675 train / 1,161 test**, and the two pools are disjoint.
- **Splits (`phase3/data_perf/`, sha256 in `manifest_scaling.json`):** nested train subsets `train_{10,30,50,60}k.jsonl` (prefixes of one seeded shuffle); `validation/hi/val_mix.jsonl` (2,000, train-pool names) and `val_heldout.jsonl` (1,500, held-out names, used for selection); and test slices:
  - in-domain: `rel_id_d23`, `num_id_d23`, `mix_id_d23`, `rel_ood_d4`, `rel_ood_d5`, `rel_ood_d6`
  - cross-domain: {speed, strength, diligence, temperature} × {`id_d23`, `ood_d4`, `ood_d5`, `ood_d6`}

  Trained depths are 2–3 and held-out depths 4/5/6. Trained domains are height, age, wealth and weight. The family mix is relation/numeric/mixed at 50/30/20. HF dataset: `Prateek-Tiwari10/LMA_phase3_data`.
- **Loss:** prompt-masked (prompt labels = −100), dynamic padding with EOS, token-weighted across micro-batches (`phase3/sft/data.py`, `train.py`).
- **Metrics** (`phase3/eval/metrics.py`, `analysis.py`, `matrix.py`, `parser.py`): exact match, chain-correct, `copy_rate` (fraction of chain entities copied from the prompt), name fidelity (in prompt / from the finetuning pool), per-depth curves, per-slice and per-family breakdowns, Wilson CIs, chance baselines, and an error taxonomy.
- **Finetune hyperparameters:** see §10b.

---

## 9. Existing eval and attention scripts

| Need | Script |
|---|---|
| Loss, PPL, BPB on pretraining splits (each token scored once, non-overlapping windows) | `lm/evaluate.py` (`score_split`, `utf8_bytes_of`) |
| BLEU-4 (sacrebleu), chrF2, ROUGE-L (own LCS), Distinct-1/2, 4-gram repetition, greedy plus T ∈ {0.5, 1.0, 1.5}, seeded | `lm/generate.py` |
| Same generation metrics on the reasoning task | `phase3/eval/generation_metrics.py` |
| Prompt-masked PPL/BPB on reasoning slices | `phase3/eval/perplexity.py` |
| Forgetting (PPL on Phase 2 test prefix) | `phase3/eval/forgetting.py` |
| Heatmaps, per-head entropy, mean attention distance, causal check | `lm/attention.py` (`collect_attention`, `head_statistics`, `plot_heatmaps`, `plot_statistics`, `verify_causal`) |
| Pre- vs post-finetune attention on reasoning prompts (64 `rel_id_d23` held-out-name prompts) | `phase3/eval/attention.py` (FT_SIZE = "60k") |
| Reasoning eval matrix | `phase3/eval/matrix.py` (reads `scaling_*/best.pt`), `run.py`, `gates.py` |

---

## 10. Final finetuned GPT checkpoint and the data and hyperparameters that produced it

**Open question for Prateek: there is no single "final" Hindi finetune.** There are four scaling runs: ft_10k, ft_30k, ft_50k and ft_60k. For the following reasons, I propose **`ft_60k` → `phase3/runs/scaling_hindi_60k/best.pt`** as the GPT-ft baseline:

- the Phase 3 attention analysis uses it (`FT_SIZE = "60k"`);
- it has the most data;
- it converged;
- `ft_10k` is documented as unconverged.

HF: `Prateek-Tiwari10/LMA_phase3` → `hindi/ft_60k/`.

| Item | Value |
|---|---|
| Base | `final_16k.pt` (step 16,125) with a **fresh** AdamW (no optimiser state carried over) |
| Train data | `data_scaling_60k/hi/train.jsonl` = `data_perf/train/hi/train_60k.jsonl` (60,000 records, sha256 `46f1c3b6…a668632`, verified locally), 5.78M tokens (3.52M supervised) |
| Val (loss) | `val.jsonl` = `val_mix.jsonl` (2,000, sha256 `350b16f2…`) |
| Val (selection, EM) | `val_heldout.jsonl` (1,500, sha256 `2d16d072…`). A fixed 500-example subset is drawn with `default_rng(1337+7)`. Greedy decoding, max 160 new tokens. |
| Name pool | Expanded: 3,675 train / 1,161 test (held-out) |
| Peak LR | **3e-4**. Swept twice: Part 2 {1e-4, 2e-4, **3e-4**, 4.5e-4, 6e-4}; perf build {2e-4, **3e-4**, 4e-4} at half an epoch. 3e-4 was chosen as the interior optimum. |
| Schedule | Warmup 3 % of **planned** steps (int(0.03·7,496) = 224), then cosine to 0.1·peak over **7,496 planned steps** (937 steps/epoch × 8 epochs) |
| Batch | 16 micro × 4 accum = 64 sequences, max_len 384, length-bucketed (window 512) |
| Optimiser, clip, precision | AdamW (0.9, 0.95), eps 1e-8, wd 0.1; clip 1.0; fp16 + GradScaler; dropout 0.1 (kept on) |
| Data order | `np.random.default_rng(seed·100003 + epoch).permutation(n)` followed by length bucketing. This is a pure function of (seed 1337, epoch), so it is **exactly reproducible**. |
| Eval and stopping | Eval every 500 steps. **Early stopping, patience 2** on held-out EM. Stopped at **step 2,000**; **best.pt = step 1,000** (epoch 1), held-out val EM **0.954**, val loss 0.00172. |
| Wall time | 591 s on one T4 |
| Headline test (from `report/RESULTS_scaling.md`) | `rel_id_d23` EM 96.8 %; in-domain d2–3 EM 95.9 / chain 93.2; d4–6 EM 30.2 / chain **0.0**; cross-domain d2–3 EM 52.5 |

I'm treating the earlier `extra/runs_superseded/sft_hindi*` runs as superseded and excluding them.

---

## 10b. Replication spec the LLaMA runs must copy

### Pretraining (copy exactly)

| Field | Value |
|---|---|
| Tokenizer | `hindi_bpe_16000.model` (md5 `e50d13a0…`), vocab 16,000, EOS 2 |
| Data | `train_16k.bin` / `val_16k.bin` / `test_16k.bin` (sha1 heads in §5) |
| Data order | `TokenStream.sample` with `np.random.default_rng(1337)`, 2 × 32 offsets per step. Same RNG consumption gives byte-identical batches. |
| Val batches | `default_rng(1338)`, 50 × 32 × 512, every 250 steps plus the final step |
| Steps / tokens | 16,125 steps / 528,384,000 tokens |
| Batch / context | 32 × 2 accum × 512 = 32,768 tokens/step |
| Schedule | warmup 750 linear → cosine → floor 0.1·peak at 16,125 (`TrainConfig.lr_at`) |
| Optimiser | AdamW (0.9, 0.95), eps 1e-8, wd 0.1 on dim ≥ 2 only; clip 1.0 |
| Precision | fp16 autocast + GradScaler, single GPU |
| Dropout | 0.1 at the same sites |
| Seeds | torch / np 1337 |
| Checkpoints | every 500 steps |
| Peak LR | **Only this differs.** Taken from the LLaMA sweep (Stage 3). |

### Finetuning (copy exactly)

Data `train_60k.jsonl` / `val_mix.jsonl` / `val_heldout.jsonl` (sha256 above), with the remaining fields copied from §10.

### Cannot be recovered exactly → documented limitations, not a reason to retrain GPT

1. **Model initialisation differs.** The architectures differ, so the torch RNG stream maps to different tensors. Unavoidable.
2. **Dropout masks differ.** Same reason.
3. **Software stack.** The Kaggle image's torch/CUDA versions at the time of the GPT run are not recorded in the logs I found. fp16 kernels may differ slightly. I'll record the versions used for LLaMA.
4. **No val BPB in the GPT log.** Only val loss is logged. The 50 val windows are deterministic (rng 1338), though, so for each of the 65 GPT points **BPB = val_loss_nats × tokens / (ln 2 × UTF-8 bytes of those exact target tokens)**. That is an exact transform on the same token set, with no checkpoint re-evaluation needed. One small caveat: the GPT loss is a mean of 50 batch means, which equals the global mean because the batches are equal-sized. I'll label these points "derived from log". As far as I can tell, intermediate GPT checkpoints were not kept (only `final_16k.pt`), so re-evaluation at intermediate steps is not possible anyway.
5. **Single GPT pretraining seed, single GPT finetune seed.** Rule 8 forbids reruns.

---

## 11. Deviations from the instruction file that need your decision

1. **LR-sweep protocol (Stage 3) vs GPT's actual sweep.** GPT's LR *was* swept with a bracket, so the "baseline untuned" limitation does not apply. But its sweep used **1,200 steps (7.44 % of budget) with warmup 150 (12.5 %)**, not the full run's 4.65 % warmup fraction, on the grid {1.5e-3, 3e-3, 4.5e-3, 6e-3}. The instruction file asks for "the same warmup fraction as the full run" and suggests a 1e-4 to 2e-3 bracket.
   - **I recommend copying GPT's sweep protocol exactly** (1,200 steps, warmup 150, same grid, extending one notch if the winner sits at an edge). That gives both architectures an identical tuning budget and procedure, which is the actual fairness rule.
2. **T4×2.** The GPT run was single-GPU. DDP would change per-step batch composition and RNG consumption, which breaks the exact-data-order replication.
   - **I recommend single-GPU training**, with the second T4 used only to run a second, independent sweep point in parallel.
3. **Which GPT-ft is the baseline.** I recommend ft_60k `best.pt`, as set out in §10.
4. **Early stopping in finetuning.** GPT-ft stopped by early stopping (best at step 1,000, stopped at 2,000), so LLaMA's step count may differ. **I recommend copying the protocol as-is**: same patience, same selection signal, and the same 7,496-step planned cosine horizon. I'll report the steps actually run. The alternative is to force LLaMA to exactly 1,000 or 2,000 steps.
5. **Instruction-file wording.** The file says "~500M tokens". The exact figure is 528,384,000 tokens seen, from a 528,404,893-token train split.
6. **Stage 5, 2× context for LLaMA.** RoPE allows this, but the causal mask buffer must not be fixed at 512. I'll build the mask on the fly in `llama.py` (the GPT code is unchanged).

---

## 12. What went wrong / open questions

- Nothing failed. All reads succeeded, and `final_16k.pt` loads strictly into the code-built model.
- Open questions: the four decisions in §11.
