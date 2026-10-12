"""Publish trained tokenizers to the Hugging Face Hub, and fetch them back.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

WHY THE HUB SITS IN THE MIDDLE
------------------------------
Training happens on Kaggle (the corpus does not fit in this laptop's RAM), but
tokenizing the corpus happens locally.  Rather than downloading model files out
of a Kaggle notebook by hand, the Hub is the handoff point:

    Kaggle:  train  ->  push_tokenizer()   ->  Hugging Face
    local:   fetch_tokenizer()  <-  Hugging Face  ->  data.pipeline.export

That also makes the tokenizers a citable artifact for the report, and means a
re-run on any machine starts from exactly the model that produced the numbers.

WHAT GETS UPLOADED
------------------
Per language, into its own subfolder of one repository:

    <lang>/<lang>_final.model     the chosen tokenizer, trained on the FULL corpus
    <lang>/<lang>_final.vocab     its vocabulary listing
    <lang>/tokenizer_final.json   what it is and how it scored
    <lang>/tokenizer_sweep.json   the sweep it was chosen from -- the justification
    <lang>/sweep_models/*         every sweep model, so the comparison is reproducible

THE TOKEN IS NEVER WRITTEN DOWN
-------------------------------
The write token is read from the environment (``HF_TOKEN``), which on Kaggle comes
from **Add-ons -> Secrets**.  It is never hard-coded, never printed, and never
committed.  A token pasted into a notebook cell is published the moment the
notebook is shared or committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

LOGGER = logging.getLogger("hf_tokenizer")

#: Default repository holding both languages' tokenizers, one subfolder each.
DEFAULT_REPO = "Prateek-Tiwari10/lma_phase1-tokenizer"


def _token() -> str:
    """Return the Hugging Face write token from the environment.

    Returns:
        The token string.

    Raises:
        SystemExit: If no token is set, with instructions rather than a traceback.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "No Hugging Face token found.\n"
            "  On Kaggle:  Add-ons -> Secrets -> add HF_TOKEN, then\n"
            "              from kaggle_secrets import UserSecretsClient\n"
            "              os.environ['HF_TOKEN'] = "
            "UserSecretsClient().get_secret('HF_TOKEN')\n"
            "  Locally:    export HF_TOKEN=hf_...   (do not commit it)"
        )
    return token


def build_model_card(languages: list[str], reports: dict) -> str:
    """Write the repository README shown on the Hub.

    Args:
        languages: Languages included.
        reports: ``{language: final_report_dict}``.

    Returns:
        Markdown model card.
    """
    rows = []
    for lang in languages:
        chosen = reports.get(lang, {}).get("chosen", {})
        rows.append(
            f"| {lang.capitalize()} | {chosen.get('model_type', '?').upper()} | "
            f"{chosen.get('vocab_size', 0):,} | "
            f"{chosen.get('tokens_per_word', 0):.4f} | "
            f"{chosen.get('chars_per_token', 0):.4f} | "
            f"{chosen.get('unk_rate', 0):.1e} |"
        )

    return f"""---
language:
- hi
license: apache-2.0
tags:
- tokenizer
- sentencepiece
- devanagari
- hindi
---

# Hindi Tokenizer

A SentencePiece tokenizer trained from scratch for ~25M-parameter Hindi
decoder-only language models.

## Chosen models

| Language | Algorithm | Vocab | Fertility (tok/word) | chars/token | UNK rate |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Each was selected from a sweep of **BPE and Unigram × 16,000 and 24,000**, then
retrained on the **complete** cleaned corpus. `tokenizer_sweep.json` in each
language folder holds the full comparison.

## Why these vocabulary sizes

Fertility improves monotonically with vocabulary, so it cannot be the only
criterion. At `d_model=384` the embedding table costs `vocab × 384` out of a ~25M
budget, and one decoder layer costs ~`12 × 384²` = 1.77M:

| Vocab | Embeddings | Share of budget | Decoder layers affordable |
|---|---:|---:|---:|
| 16,000 | 6.14M | 25% | ~10.7 |
| 24,000 | 9.22M | 37% | ~8.9 |
| 64,000 | 24.58M | 98% | **~0** |

A vocabulary much above ~32k spends the model on its own embedding table.

## UNK rate is zero by construction

`byte_fallback=True`: any character the vocabulary cannot cover decomposes into
UTF-8 byte tokens (ids 3–258). Nothing is ever discarded.

## ⚠️ Input must be normalized first

These were trained on text normalized by the project's `data/pipeline/normalize.py`,
which folds chandrabindu (ँ) to anusvara (ं) for Hindi.

Feeding un-normalized text to the Hindi model triggers byte fallback and inflates
the token count. Always normalize with the matching language setting before
encoding.

## Usage

```python
import sentencepiece as spm
from huggingface_hub import hf_hub_download

path = hf_hub_download("{DEFAULT_REPO}", "hindi/hindi_final.model")
sp = spm.SentencePieceProcessor(model_file=path)
print(sp.encode("प्रधानमंत्री ने कहा कि सरकार काम कर रही है।", out_type=str))
```

## Files

```
hindi/   hindi_final.model · hindi_final.vocab
         tokenizer_final.json · tokenizer_sweep.json · sweep_models/
```
"""


def push_tokenizer(language: str, tokenizer_dir: Path, sweep_report: Path,
                   final_report: Path, repo_id: str = DEFAULT_REPO,
                   staging: Path | None = None) -> None:
    """Upload one language's tokenizers and reports to the Hub.

    Args:
        language: ``"hindi"``.
        tokenizer_dir: Directory holding the ``.model``/``.vocab`` files.
        sweep_report: The sweep JSON (the justification).
        final_report: The final-model JSON.
        repo_id: Target repository.
        staging: Scratch directory for assembling the upload.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=_token())
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    staging = staging or Path("/tmp/hf_stage") / language
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "sweep_models").mkdir(parents=True)

    final = json.loads(final_report.read_text())
    chosen = final["chosen"]
    src = Path(chosen["model_path"])

    # The chosen model gets a stable, language-scoped name so downstream code
    # never has to know which algorithm or vocabulary size won.
    shutil.copy(src, staging / f"{language}_final.model")
    vocab = src.with_suffix(".vocab")
    if vocab.exists():
        shutil.copy(vocab, staging / f"{language}_final.vocab")

    shutil.copy(sweep_report, staging / "tokenizer_sweep.json")
    shutil.copy(final_report, staging / "tokenizer_final.json")

    # Every sweep model travels too, so the comparison can be re-verified rather
    # than merely read.
    for path in sorted(Path(tokenizer_dir).glob("*.model")):
        if path.resolve() != src.resolve():
            shutil.copy(path, staging / "sweep_models" / path.name)

    LOGGER.info("%s: uploading %d files to %s/%s",
                language, len(list(staging.rglob("*"))), repo_id, language)
    api.upload_folder(folder_path=str(staging), path_in_repo=language,
                      repo_id=repo_id, repo_type="model",
                      commit_message=f"Add {language} tokenizer "
                                     f"({chosen['model_type']} {chosen['vocab_size']})")
    LOGGER.info("%s: done -> https://huggingface.co/%s/tree/main/%s",
                language, repo_id, language)


def push_model_card(languages: list[str], report_dir: Path,
                    repo_id: str = DEFAULT_REPO) -> None:
    """Generate and upload the repository README.

    Args:
        languages: Languages to document.
        report_dir: Where the ``tokenizer_final_*.json`` files live.
        repo_id: Target repository.
    """
    from huggingface_hub import HfApi

    reports = {}
    for lang in languages:
        path = report_dir / f"tokenizer_final_{lang}.json"
        if path.exists():
            reports[lang] = json.loads(path.read_text())

    card = Path("/tmp/hf_README.md")
    card.write_text(build_model_card(languages, reports), encoding="utf-8")
    HfApi(token=_token()).upload_file(
        path_or_fileobj=str(card), path_in_repo="README.md",
        repo_id=repo_id, repo_type="model",
        commit_message="Update model card")
    LOGGER.info("model card -> https://huggingface.co/%s", repo_id)


def fetch_tokenizer(language: str, repo_id: str = DEFAULT_REPO,
                    out_dir: Path | None = None,
                    report_dir: Path = Path("report")) -> Path:
    """Download one language's final tokenizer and reports from the Hub.

    This is the local side of the handoff: it recreates exactly what
    ``data.pipeline.export`` expects, so tokenizing the corpus locally needs no manual
    file shuffling.

    Args:
        language: ``"hindi"``.
        repo_id: Source repository.
        out_dir: Where to put the model; defaults to ``<lang>/tokenizer/``.
        report_dir: Where to put the reports.

    Returns:
        Path to the downloaded ``.model`` file.
    """
    from huggingface_hub import hf_hub_download

    out_dir = out_dir or Path(language) / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Public repo: a token is optional for reading.
    token = os.environ.get("HF_TOKEN")

    model_path = out_dir / f"{language}_final.model"
    for remote, local in (
        (f"{language}/{language}_final.model", model_path),
        (f"{language}/{language}_final.vocab", out_dir / f"{language}_final.vocab"),
        (f"{language}/tokenizer_sweep.json", report_dir / f"tokenizer_sweep_{language}.json"),
        (f"{language}/tokenizer_final.json", report_dir / f"tokenizer_final_{language}.json"),
    ):
        try:
            cached = hf_hub_download(repo_id, remote, repo_type="model", token=token)
            shutil.copy(cached, local)
            LOGGER.info("  %s -> %s", remote, local)
        except Exception as exc:
            if remote.endswith(".model"):
                raise
            LOGGER.warning("  %s unavailable (%s)", remote, type(exc).__name__)

    # The final report records the path the model had ON KAGGLE. Rewrite it to
    # the local path, or export.py will look for a file that does not exist here.
    final_report = report_dir / f"tokenizer_final_{language}.json"
    if final_report.exists():
        data = json.loads(final_report.read_text())
        data["chosen"]["model_path"] = str(model_path)
        data["chosen"]["fetched_from"] = f"{repo_id}/{language}"
        final_report.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        LOGGER.info("  rewrote model_path -> %s", model_path)

    return model_path


def main() -> None:
    """Push or fetch tokenizers."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("push", "fetch"))
    parser.add_argument("--languages", nargs="+", default=["hindi"])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    if args.action == "push":
        for lang in args.languages:
            push_tokenizer(
                language=lang,
                tokenizer_dir=Path(lang) / "tokenizer",
                sweep_report=args.report_dir / f"tokenizer_sweep_{lang}.json",
                final_report=args.report_dir / f"tokenizer_final_{lang}.json",
                repo_id=args.repo)
        push_model_card(args.languages, args.report_dir, args.repo)
    else:
        for lang in args.languages:
            LOGGER.info("fetching %s", lang)
            fetch_tokenizer(lang, args.repo, report_dir=args.report_dir)
        LOGGER.info("done. Next: python -m data.pipeline.export --language <lang>")


if __name__ == "__main__":
    main()
