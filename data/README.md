# Data — how the Hindi corpus and token files were built

> **Reference code: the token files were built once and are not regenerated.**
> `data/pipeline/` is the code that built them. It is kept so the repository shows
> every step, and it compiles (`python -m py_compile`), but nothing in this study
> runs it. Training and evaluation read only the token bins listed under *Token
> files* below.

| Folder | What it is | Run in this study? |
|---|---|---|
| `data/pipeline/` | Corpus collection, cleaning, normalisation, dedup, language-ID filter, splits, tokenizer training, export | **No**: reference only |
| `data/token_stream.py` | Memory-mapped reader and seeded batch sampler for the token bins | Yes |
| `data/sft_data.py` | Prompt-masked tokenisation and batching for the reasoning finetune | Yes |
| `data/reasoning/` | Generator for the synthetic reasoning dataset, its configs, name pools and sha256 manifest | Reference; data already generated |
| `data/overlap.py` | Train vs held-out near-duplicate check | Reference |

---

## 1. Sources: manual crawl vs downloaded

About **21 % of tokens come from our own crawl** (`source_type="manual"`); the rest
were downloaded from public corpora.

| Type | Source | Code |
|---|---|---|
| Manual crawl | 10 Hindi news sites, discovered from their sitemaps: Dainik Jagran, Amar Ujala, Live Hindustan, Dainik Bhaskar, Prabhat Khabar, Patrika, ABP Live, Navbharat Times, Aaj Tak, BBC Hindi. The crawl was polite (1 req/s per host) and respected `robots.txt`. | `discover.py` → `frontier.py` (SQLite URL queue) → `crawl.py` / `fetch.py`; site list in `sources_hindi.json` |
| Downloaded | `ai4bharat/sangraha` verified Hindi split (weight 0.85) and Hindi Wikipedia `20231101.hi` (weight 0.15), streamed shard by shard | `public_corpus.py` |

Manual share per split (from `reports/data/export_16k_hindi.json`): train 20.8 %, val 20.8 %, test 21.1 %.

## 2. Cleaning (`clean.py` orchestrates stages 1–5)

1. **Unicode normalisation** (`normalize.py`). Steps:
   - NFC;
   - nukta forms unified;
   - ZWJ/ZWNJ stripped;
   - Devanagari digits mapped to ASCII;
   - punctuation variants unified;
   - Vedic marks stripped;
   - **chandrabindu folded to anusvara** (the Hindi setting).
2. **Language-ID filter** (`lid.py`, trained by `train_lid.py`; model in `pipeline/models/lid_hi_ne.json`). This is the **filter used to keep only Hindi documents**: a character 4-gram Hindi-vs-Nepali classifier plus function-word markers. Documents not confidently labelled Hindi are dropped.
3. **Quality filter** (`quality.py`). Rejects:
   - boilerplate;
   - too short or too few words;
   - low Devanagari ratio (English or romanised text);
   - repetitive bigrams;
   - digit- or punctuation-heavy pages.
4. **Sentence-level cleanup**. Removes repeated boilerplate sentences inside otherwise good documents.
5. **Split assignment**. Held-out val/test documents are separated *before* dedup, so the training split can be decontaminated against them.
6. **Deduplication** (`dedup.py`). Exact hashes plus MinHash-LSH (5-shingles, 128 permutations, 16 bands); near-duplicates are verified with Jaccard.
7. **Audit** (`audit.py`). Post-clean sanity statistics.

Funnel (from `reports/data/clean_hindi.json`):

| Stage | Documents |
|---|---:|
| Input (41 shards) | 1,712,057 |
| Dropped by language-ID filter | 8,283 |
| Dropped by quality filter | 219,836 |
| Kept after cleaning | 1,483,938 |
| Removed by dedup (20,627 exact, 7,335 near-duplicates) | 27,962 |
| **After dedup** | **1,455,976** |

The exported splits were then capped near the 500M-token training target:

| Split | Documents | Tokens (16K BPE) |
|---|---:|---:|
| train | 975,528 | **528,404,893** |
| val | 66,347 | 35,779,979 |
| test | 66,330 | 35,903,951 |

Train vs held-out overlap (`reports/data/split_overlap.json`): **0 exact duplicates**, and **2 near-duplicates** in a 48,776-document MinHash sample. Both are the same templated movie-listing disclaimer.

Figures: `reports/data/figures/` (cleaning funnel, corpus composition, document lengths, quality rejections).

## 3. Tokenizer (`tokenizer.py`, `eval_tokenizer.py`)

- SentencePiece **BPE, 16,000 vocab**, `byte_fallback=True` (so the UNK rate is 0), trained on the training split only.
- Chosen from a sweep of BPE/Unigram × 16K/24K. The 16K vocabulary keeps the embedding table to 6.14M parameters, about 24 % of a 25.9M model.
- Measured on held-out text: **4.09 characters per token, 1.276 tokens per word**.
- EOS id 2 (`</s>`) separates documents in the token stream.
- Files: `tokenizer/hindi_bpe_16000.model` / `.vocab` (md5 `e50d13a005839869fb9b466ef89d2aa7`). Also published at HF `Prateek-Tiwari10/lma_phase1-tokenizer`.
- `tokenizer/hindi_bpe_24000.*` is the 24K tokenizer of an earlier baseline. It isn't used in this study and is kept for reference.

## 4. Token files

The bins are uint16 token streams with a JSON sidecar each, written by `export.py`. They are **not in git**: they live under `artifacts/data/bin/` (gitignored).

| File | Tokens | sha256 |
|---|---:|---|
| `train_16k.bin` | 528,404,893 | `1bfa57d73456e5c0bec978243216b06124cc9bfbe7f64ae1e0992aeb03488cde` |
| `val_16k.bin` | 35,779,979 | `b2d67d5d96d019e70456e79acfcc07d6d32d73b85e2d9ff0249a7b12e6ae513b` |
| `test_16k.bin` | 35,903,951 | `e59681ddbb681aa807a9bb93ce8005bdd12c70f56d94a0e3872681b76845cce3` |

Also listed in `reports/data/bin_checksums_16k.txt`. Each sidecar records a sha1 of the first 64 MB, the token count, the max id and the tokenizer md5. `data/token_stream.py` checks these before training.

**Where they are stored:**
- Locally: `artifacts/data/bin/`.
- On Kaggle: the output of kernel `prateek99491/lma-phase-2-export-bins-16k` (account `prateek99491`). That kernel rebuilds byte-identical bins; the sha256 values above match.
- Drive / HF mirror: *link to be added by Prateek*.

**How training reads them:** `TokenStream.sample` draws uniformly random 512-token windows with `np.random.default_rng(1337)`, so the batch order is a pure function of the seed. Validation always scores the same 50 × 32 windows, drawn with seed 1338.

## 5. Reasoning finetune data (`data/reasoning/`)

Synthetic transitive-comparison problems in Hindi, with chain-of-thought targets.

- **Generator:** `generator.py`, run by `run.py`, gated by `checks.py` G1–G8. The configuration is `configs/hi_perf.yaml`, the "perf" build with the expanded name pools. Master seed 20260910, subset seed 20260912.
- **Name pools:**
  - training pool: 3,675 names after filtering (`vocab/hi_names_train_perf.txt` has 3,698 lines);
  - held-out test pool: 1,161 names after filtering (1,184 lines), disjoint from the training pool;
  - the original 243-name pool, whose memorisation motivated the rebuild, is in `vocab/hi_names_train.txt`.
- **Splits:**
  - `train_{10,30,50,60}k.jsonl`: nested prefixes of one seeded shuffle;
  - `val_mix.jsonl`: training-pool names;
  - `val_heldout.jsonl`: held-out names, used for model selection;
  - 22 test slices: in-domain depths 2–3 and held-out depths 4/5/6, plus 4 unseen domains × 4 depth cuts.
- **Location:** generated files are under `artifacts/finetune_data/` (gitignored), with sha256 values in `data/reasoning/manifest_scaling.json`. They are published at HF `Prateek-Tiwari10/LMA_phase3_data`.
- **What the GPT-ft baseline used:** `train/hi/train_60k.jsonl` (sha256 `46f1c3b6…a668632`), `validation/hi/val_mix.jsonl` and `validation/hi/val_heldout.jsonl`.
