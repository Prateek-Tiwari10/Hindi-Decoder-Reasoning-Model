"""Phase 3 — perplexity and bits-per-byte on the reasoning task.

WHAT IS SCORED, AND WHY IT IS PROMPT-MASKED
--------------------------------------------
Phase 2 reported perplexity over free text.  The equivalent here is perplexity
over the *reasoning chain*: the prompt is given, so scoring the model on its
ability to predict the question it was handed would dilute the number with
tokens it is not responsible for.  Positions inside the prompt therefore carry
the label ``-100`` and drop out of the loss -- exactly the mask the model was
finetuned under, so the number is comparable with the training curve.

THE PRETRAINED ROW IS THE POINT
-------------------------------
Exact match puts the pretrained base at 0/500, which is true but uninformative:
it says the base never emits the answer format, not how far it was from the
task.  Perplexity is continuous and says how surprised each model is by the
gold chain, so the pretrained row measures the distance finetuning actually
closed.

BITS PER BYTE
-------------
Perplexity is per token and therefore not comparable across tokenizers.
Bits-per-byte divides the same total by the UTF-8 length of the reference text,
which is tokenizer-independent -- the only honest way to put two models
on one axis.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch

from model.config import ModelConfig, Paths
from model import build_model
from data.sft_data import build_example, collate

LOGGER = logging.getLogger("eval.reasoning_perplexity")

LANGS = (("hindi", "hi"),)
SIZES = ("10k", "30k", "50k", "60k")

GROUPS = {
    "in-domain d2-3": ("rel_id_d23", "num_id_d23", "mix_id_d23"),
    "in-domain d4-6": ("rel_ood_d4", "rel_ood_d5", "rel_ood_d6"),
    "cross-domain d2-3": ("xdom_speed_id_d23", "xdom_strength_id_d23",
                          "xdom_diligence_id_d23"),
}

#: Matches the generation protocol in the scaling matrix.
PER_SLICE = 500

#: The logits tensor is (B, T, 16000) and dominates memory, so the batch is
#: sized for a 4 GB card rather than for throughput.  Perplexity is one forward
#: pass per example either way.
BATCH = 8


def corpus_path(root: Path, code: str, slice_tag: str) -> Path:
    """Resolve a slice file; cross-domain slices nest under the domain name."""
    base = Paths.for_variant("hindi", root).finetune_data / "test" / code
    direct = base / "in_domain" / f"{slice_tag}.jsonl"
    if direct.exists():
        return direct
    matches = sorted(base.rglob(f"{slice_tag}.jsonl"))
    if not matches:
        raise SystemExit(f"no corpus file for {slice_tag!r}")
    return matches[0]


def load_group(root: Path, code: str, slices, tokenizer, eos_id: int):
    """Tokenize and mask every record in a slice group."""
    examples, gold_bytes = [], 0
    for slice_tag in slices:
        with corpus_path(root, code, slice_tag).open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i >= PER_SLICE:
                    break
                record = json.loads(line)
                example, _ = build_example(record, tokenizer, eos_id)
                examples.append(example)
                gold_bytes += len(record["completion"].encode("utf-8"))
    return examples, gold_bytes


def score(model, examples, pad_id: int, device: str, gold_bytes: int,
          batch_size: int = BATCH) -> dict:
    """Total prompt-masked loss over a group, as perplexity and bits-per-byte."""
    total_nats, total_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            x, y, n_supervised = collate(batch, pad_id, device)
            # fp32 throughout.  Autocast would halve the logits tensor and let
            # this run on a smaller card, but perplexity is a reported number
            # and fp16 rounding in the softmax is not worth the memory.
            # The batch is what keeps it inside 4 GB.
            _, loss, _ = model(x, targets=y)
            total_nats += loss.item() * n_supervised
            total_tokens += n_supervised

    mean_nats = total_nats / total_tokens
    return {
        "n_examples": len(examples),
        "supervised_tokens": total_tokens,
        "utf8_bytes": gold_bytes,
        "mean_ce_nats": mean_nats,
        "perplexity": math.exp(mean_nats),
        "bits_per_token": mean_nats / math.log(2),
        "bits_per_byte": (total_nats / math.log(2)) / gold_bytes,
    }


def load_model(path: Path, device: str):
    """Build a model from its own recorded config."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(state["model_config"])
    model = build_model(config, device=device, arch=state.get("arch", "gpt"))
    model.load_state_dict(state["model"])
    model.eval()
    return model


def parse_model_args(items: list[str]) -> list[tuple[str, Path]]:
    """Parse repeated ``TAG=PATH`` command-line items.

    Args:
        items: Raw ``--model`` values.

    Returns:
        ``(tag, path)`` pairs.
    """
    pairs = []
    for item in items:
        tag, _, path = item.partition("=")
        if not tag or not path:
            raise SystemExit(f"--model expects TAG=PATH, got {item!r}")
        pairs.append((tag, Path(path)))
    return pairs


def main() -> None:
    """Score the pretrained base and all four finetunes, both languages."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_finetune/perplexity.json"))
    parser.add_argument("--model", action="append", default=[],
                        metavar="TAG=PATH",
                        help="extra checkpoint to score (repeatable); the "
                             "GPT baseline pretrained + ft_60k are always scored")
    args = parser.parse_args()

    import sentencepiece as spm

    root = args.root.resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload: dict = {}

    for language, code in LANGS:
        paths = Paths.for_variant(language, root, "16k")
        tokenizer = spm.SentencePieceProcessor(model_file=str(paths.tokenizer))
        eos_id, pad_id = tokenizer.eos_id(), tokenizer.eos_id()

        groups = {name: load_group(root, code, slices, tokenizer, eos_id)
                  for name, slices in GROUPS.items()}

        checkpoints = {"pretrained": paths.final_model,
                       "ft_60k": paths.gpt_finetuned}
        checkpoints.update(dict(parse_model_args(args.model)))

        payload[language] = {}
        for tag, path in checkpoints.items():
            if not path.exists():
                LOGGER.warning("skipping %s: %s missing", tag, path)
                continue
            model = load_model(path, device)
            payload[language][tag] = {}
            for name, (examples, gold_bytes) in groups.items():
                cell = score(model, examples, pad_id, device, gold_bytes,
                             args.batch)
                payload[language][tag][name] = cell
                LOGGER.info("%s %-10s %-18s ppl %8.2f  bpb %.4f  CE %.4f",
                            code, tag, name, cell["perplexity"],
                            cell["bits_per_byte"], cell["mean_ce_nats"])
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
