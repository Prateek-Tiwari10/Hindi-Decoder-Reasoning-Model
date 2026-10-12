# Tokenizer

| File | What |
|---|---|
| `hindi_bpe_16000.model` / `.vocab` | **The tokenizer every model in this study uses.** SentencePiece BPE, 16,000 vocab, byte fallback, EOS id 2. md5 `e50d13a005839869fb9b466ef89d2aa7`. |
| `hindi_bpe_24000.model` / `.vocab` | 24K BPE from an earlier baseline; not used here, kept for reference. |

Also published at Hugging Face `Prateek-Tiwari10/lma_phase1-tokenizer`.  How it
was trained: `data/README.md` §3.  Input text must be normalised with
`data/pipeline/normalize.py` (Hindi settings) before encoding, exactly as the
training corpus was.
