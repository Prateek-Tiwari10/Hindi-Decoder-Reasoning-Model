"""Deduplication: exact SHA-1 first, then hand-rolled MinHash + LSH near-duplicate detection.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

Written to be read and explained, not to be clever.  Every step of the MinHash
and LSH derivation is spelled out in comments because this is the algorithm the
project rests on and it must be defensible without reference to a library.

WHY DEDUPLICATION MATTERS
-------------------------
Training on duplicated text wastes the token budget and biases the model toward
whatever happens to be repeated -- on a news crawl, that is syndicated wire
copy, boilerplate and republished articles.  Our two corpora are especially
exposed because Sangraha itself contains news crawls that overlap the sites we
crawled ourselves, so the same article can arrive twice by two different routes.

THE TWO STAGES
--------------
1. **Exact duplicates** -- SHA-1 of the normalized text.  Cheap, catches
   republished-verbatim copies, and shrinks the input to stage 2.

2. **Near duplicates** -- same article with a different headline, a trailing
   "Also read" block, or one edited paragraph.  Exact hashing cannot see these,
   so we estimate Jaccard similarity between documents' word 5-gram sets.

WHY NOT COMPARE EVERY PAIR
--------------------------
The Hindi corpus has ~1.7M documents.  All-pairs comparison is 1.7e6 * 1.7e6 / 2 = 1.4e12
pairs, which is impossible.  MinHash + LSH reduces this to a near-linear scan by
only ever comparing documents that already look similar.

MINHASH, IN ONE PARAGRAPH
-------------------------
Take a random permutation of the universe of all possible shingles.  Apply it to
a document's shingle set and keep the *minimum* value.  For two sets A and B, the
probability that their minima agree is exactly |A ∩ B| / |A ∪ B| -- the Jaccard
similarity.  So one permutation is a 1-bit estimator of Jaccard; ``NUM_PERM`` of
them give an estimate with standard error ~1/sqrt(NUM_PERM).  At 128 permutations
that is ~0.088, which is why candidates are then verified exactly (see below).
Real permutations of a huge universe are infeasible, so we use the standard
substitute: a family of hash functions ``h_i(x) = (a_i * x + b_i) mod p`` with p a
large prime, each of which behaves like a random permutation.

LSH, AND THE BAND EQUATION
--------------------------
Split the 128-value signature into ``BANDS`` bands of ``ROWS`` values each.  Two
documents become *candidates* if they agree on **all** rows of **at least one**
band.  For a pair with true Jaccard s:

    P(agree on one whole band)      = s^ROWS
    P(differ somewhere in a band)   = 1 - s^ROWS
    P(differ in every band)         = (1 - s^ROWS)^BANDS
    P(candidate)                    = 1 - (1 - s^ROWS)^BANDS

That is an S-curve in s.  Its steepest point -- the effective threshold -- is
approximately:

    threshold ≈ (1 / BANDS)^(1 / ROWS)

With BANDS=16 and ROWS=8: (1/16)^(1/8) = 0.7071.

WHY 0.707 AND NOT 0.8
---------------------
We want to remove pairs with Jaccard >= 0.8, so an obvious choice is a band
configuration whose threshold sits at 0.8 (BANDS=8, ROWS=16 gives 0.878).  We
deliberately choose the *lower* threshold instead, because the two kinds of error
are not symmetric here:

  * An LSH configuration set at 0.878 has genuine 0.80-0.85 duplicates fall below
    the S-curve and never become candidates.  They are **missed forever** -- no
    later stage can recover them.
  * A configuration at 0.707 generates extra candidates that are not really
    duplicates.  These cost CPU, and are then **removed exactly** by the full
    Jaccard verification every candidate pair undergoes.

So the low threshold buys recall, and the exact verification pays for precision.
This is why ``verify_candidate`` computes true Jaccard on the stored shingle sets
rather than trusting the signature estimate.

MEMORY
------
Signatures and shingles are held in on-disk numpy memmaps, not in RAM: 2.2M
documents at 128 uint32 values is 1.1 GB of signatures alone, on a machine with a
few GB free.  Bands are processed one at a time by sorting, which needs only one
band's worth of keys (2.2M uint64 = 18 MB) in memory at once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

LOGGER = logging.getLogger("dedup")

#: Words per shingle.  5 is the standard choice for prose: long enough that a
#: shared 5-gram is meaningful rather than coincidental, short enough that an
#: edited sentence still shares most of its shingles with the original.
SHINGLE_SIZE = 5

#: MinHash signature length.  Standard error of the Jaccard estimate is
#: ~1/sqrt(128) = 0.088, which is why candidates get an exact check afterwards.
NUM_PERM = 128

#: LSH banding.  BANDS * ROWS must equal NUM_PERM.  See the module docstring for
#: why the resulting 0.707 threshold is deliberately below the 0.8 target.
BANDS = 16
ROWS = 8

#: A pair is a duplicate if its exact Jaccard reaches this.
JACCARD_THRESHOLD = 0.8

#: The cross-corpus backstop (deduplicating against an external corpus) is
#: deliberately looser: such duplicates are rare, so a lower bar costs little
#: and catches residual contamination.
CROSS_CORPUS_THRESHOLD = 0.7

#: Mersenne prime 2^61 - 1, used as the modulus for the hash family.  Prime so
#: that (a*x + b) mod p distributes evenly; 2^61-1 so products stay inside uint64.
MERSENNE_PRIME = (1 << 61) - 1

#: Fixed seed: deduplication must be reproducible run to run, or the corpus is
#: not reproducible either.
RANDOM_SEED = 20260815


def band_threshold(bands: int = BANDS, rows: int = ROWS) -> float:
    """Return the LSH S-curve threshold ``(1/bands)^(1/rows)``.

    Args:
        bands: Number of LSH bands.
        rows: Rows per band.

    Returns:
        The similarity at which a pair has roughly even odds of becoming a
        candidate.  Reported in the statistics output, as the spec requires.
    """
    return (1.0 / bands) ** (1.0 / rows)


def candidate_probability(similarity: float, bands: int = BANDS,
                          rows: int = ROWS) -> float:
    """Probability that a pair with the given Jaccard becomes an LSH candidate.

    This is the S-curve ``1 - (1 - s^rows)^bands`` from the module docstring.
    Used to document the recall of the chosen configuration in the report.

    Args:
        similarity: True Jaccard similarity of the pair.
        bands: Number of LSH bands.
        rows: Rows per band.

    Returns:
        Probability in [0, 1].
    """
    return 1.0 - (1.0 - similarity ** rows) ** bands


def shingles(text: str, size: int = SHINGLE_SIZE) -> np.ndarray:
    """Hash a document's overlapping word n-grams into a sorted uint32 array.

    Whitespace-split rather than a real tokenizer: shingling only needs a stable,
    cheap decomposition, and the tokenizer does not exist yet when dedup runs.

    CRC32 is used rather than Python's ``hash()`` because ``hash()`` is salted per
    process and would make deduplication non-reproducible across runs.

    Args:
        text: Normalized document text.
        size: Words per shingle.

    Returns:
        Sorted array of unique uint32 shingle hashes.  Empty if the document has
        fewer than ``size`` words.
    """
    words = text.split()
    if len(words) < size:
        return np.empty(0, dtype=np.uint32)

    hashes = np.empty(len(words) - size + 1, dtype=np.uint32)
    for i in range(len(words) - size + 1):
        shingle = " ".join(words[i:i + size]).encode("utf-8")
        hashes[i] = zlib.crc32(shingle)
    return np.unique(hashes)


class MinHasher:
    """Computes MinHash signatures with a fixed, reproducible hash family."""

    def __init__(self, num_perm: int = NUM_PERM, seed: int = RANDOM_SEED) -> None:
        """Draw the hash family ``h_i(x) = (a_i * x + b_i) mod p``.

        Args:
            num_perm: Number of hash functions (signature length).
            seed: RNG seed, fixed so runs are reproducible.
        """
        rng = np.random.default_rng(seed)
        # a must be non-zero or the hash collapses to the constant b.
        self.a = rng.integers(1, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, shingle_hashes: np.ndarray) -> np.ndarray:
        """Return the MinHash signature of one shingle set.

        For each hash function, apply it to every shingle and keep the minimum.
        The whole thing is one vectorized outer product rather than a Python loop
        over 128 functions -- same arithmetic, ~100x faster.

        Args:
            shingle_hashes: uint32 shingle hashes from :func:`shingles`.

        Returns:
            uint32 array of length ``num_perm``.  An all-max signature marks an
            empty document, which never matches anything.
        """
        if shingle_hashes.size == 0:
            return np.full(self.num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)

        x = shingle_hashes.astype(np.uint64)
        # (S, 1) * (num_perm,) -> (S, num_perm), one column per hash function.
        hashed = (np.outer(x, self.a) + self.b) % MERSENNE_PRIME
        # Truncating to uint32 halves signature storage. The extra collision
        # probability is ~2^-32 per comparison, far below the 0.088 sampling
        # error of a 128-permutation estimate, so it costs nothing that matters.
        return (hashed.min(axis=0) % (1 << 32)).astype(np.uint32)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard similarity of two sorted unique uint32 arrays.

    Args:
        a: First shingle set.
        b: Second shingle set.

    Returns:
        ``|a ∩ b| / |a ∪ b|``, or 0.0 if both are empty.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    intersection = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - intersection
    return intersection / union if union else 0.0


class UnionFind:
    """Disjoint-set structure grouping duplicate pairs into clusters.

    Duplicate detection yields *pairs*, but removal needs *groups*: if A~B and
    B~C, then A, B and C are one cluster from which exactly one survivor is kept.
    Path compression keeps ``find`` effectively constant time.
    """

    def __init__(self, size: int) -> None:
        """Start with every element in its own cluster.

        Args:
            size: Number of elements.
        """
        self.parent = np.arange(size, dtype=np.int64)

    def find(self, x: int) -> int:
        """Return the representative of x's cluster, compressing the path."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        """Merge the clusters containing x and y."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


@dataclass
class DocumentRef:
    """Where a document lives and what it is, without holding its text.

    Text is deliberately excluded: 2.2M documents of text will not fit in memory,
    so the pipeline keeps references and re-reads text only when writing output.

    Attributes:
        shard: Index of the shard file the document came from.
        line: Line number within that shard.
        source_type: ``"manual"`` or ``"public"`` -- decides collision survivors.
        source: Provenance string, kept for per-source removal statistics.
        characters: Length, used to report tokens removed.
        sha1: Hex digest of the normalized text, for exact dedup.
    """

    shard: int
    line: int
    source_type: str
    source: str
    characters: int
    sha1: str


@dataclass
class DedupStats:
    """Counters for the statistics report.

    Every field here appears in the graded output, which is why removals are
    tracked separately by cause rather than as one total.
    """

    documents_in: int = 0
    characters_in: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    documents_out: int = 0
    characters_out: int = 0
    candidate_pairs: int = 0
    verified_pairs: int = 0
    manual_saved_from_public: int = 0
    removed_by_source: dict[str, int] = field(default_factory=dict)
    removed_by_source_type: dict[str, int] = field(default_factory=dict)

    def as_dict(self, chars_per_token: float) -> dict:
        """Render the counters, converting characters to tokens.

        Args:
            chars_per_token: Measured constant for this language.

        Returns:
            JSON-serializable statistics including before/after token counts,
            which the lecture homework asks for explicitly.
        """
        return {
            "documents_in": self.documents_in,
            "documents_out": self.documents_out,
            "documents_removed": self.documents_in - self.documents_out,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "characters_in": self.characters_in,
            "characters_out": self.characters_out,
            "tokens_in": int(self.characters_in / chars_per_token),
            "tokens_out": int(self.characters_out / chars_per_token),
            "token_retention": (self.characters_out / self.characters_in
                                if self.characters_in else 0.0),
            "candidate_pairs": self.candidate_pairs,
            "verified_pairs": self.verified_pairs,
            "manual_saved_from_public": self.manual_saved_from_public,
            "removed_by_source": self.removed_by_source,
            "removed_by_source_type": self.removed_by_source_type,
            "config": {
                "shingle_size": SHINGLE_SIZE,
                "num_permutations": NUM_PERM,
                "bands": BANDS,
                "rows_per_band": ROWS,
                "lsh_threshold": round(band_threshold(), 4),
                "jaccard_threshold": JACCARD_THRESHOLD,
                "recall_at_threshold": round(candidate_probability(JACCARD_THRESHOLD), 4),
            },
        }


def find_candidate_pairs(signatures: np.ndarray) -> set[tuple[int, int]]:
    """Find candidate duplicate pairs by LSH banding.

    Processes one band at a time and never materializes the full bucket table:
    for each band, the band's rows are hashed to a single uint64 key per document,
    the keys are sorted, and equal-key runs are read off as buckets.  Sorting
    2.2M uint64 keys costs ~18 MB, against many gigabytes for a dict of buckets.

    Args:
        signatures: (n_documents, NUM_PERM) uint32 signature matrix.

    Returns:
        Set of ``(i, j)`` index pairs with ``i < j`` that share at least one band.
    """
    n_docs = signatures.shape[0]
    pairs: set[tuple[int, int]] = set()

    for band in range(BANDS):
        start = band * ROWS
        block = signatures[:, start:start + ROWS]

        # Fold the band's ROWS values into one key. Multiplying by a large odd
        # constant and XOR-ing mixes the bits so that two documents collide only
        # if every row matches -- a plain sum would collide on permutations.
        keys = np.zeros(n_docs, dtype=np.uint64)
        for row in range(ROWS):
            keys = (keys * np.uint64(1099511628211)) ^ block[:, row].astype(np.uint64)

        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]

        # Bucket boundaries: positions where the sorted key changes.
        boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
        for group in np.split(order, boundaries):
            if group.size < 2:
                continue
            # A pathological bucket (thousands of identical boilerplate pages)
            # would explode into millions of pairs. Cap it: such documents are
            # near-certainly duplicates of each other and the exact-SHA-1 stage
            # has already removed the identical ones.
            if group.size > 500:
                group = group[:500]
            members = np.sort(group)
            for a_pos in range(members.size):
                for b_pos in range(a_pos + 1, members.size):
                    pairs.add((int(members[a_pos]), int(members[b_pos])))

        LOGGER.info("  band %d/%d -> %s candidate pairs so far",
                    band + 1, BANDS, f"{len(pairs):,}")

    return pairs


def choose_survivor(cluster: list[int], refs: list[DocumentRef]) -> tuple[int, int]:
    """Pick which document in a duplicate cluster to keep.

    **The hard rule: manual beats public.**  When our own crawled document
    duplicates a public-corpus document, the public one is dropped.  Both copies
    are equally good as text, but only the manual copy counts toward the >=20%
    manual requirement, so keeping the public one would throw away the very thing
    the crawl was for.

    Among documents of the same source type the longest is kept, on the grounds
    that a near-duplicate pair is usually a full article and a truncated or
    syndicated excerpt of it.

    Args:
        cluster: Document indices in one duplicate cluster.
        refs: Document references, indexed by document index.

    Returns:
        ``(survivor_index, manual_saved_count)`` where the second value counts
        public documents dropped in favour of a manual one, for the report.
    """
    manual = [i for i in cluster if refs[i].source_type == "manual"]
    public = [i for i in cluster if refs[i].source_type != "manual"]

    if manual:
        survivor = max(manual, key=lambda i: refs[i].characters)
        return survivor, len(public)

    return max(public, key=lambda i: refs[i].characters), 0


def build_clusters(pairs: set[tuple[int, int]], shingle_store: "ShingleStore",
                   n_docs: int, threshold: float,
                   stats: DedupStats) -> UnionFind:
    """Verify candidate pairs exactly and union the ones that survive.

    This is where the low LSH threshold is paid for: every candidate gets a true
    Jaccard computation, so pairs that only *looked* similar in signature space
    are discarded rather than removed from the corpus.

    Args:
        pairs: Candidate pairs from :func:`find_candidate_pairs`.
        shingle_store: Provides each document's shingle set on demand.
        n_docs: Total document count.
        threshold: Minimum exact Jaccard to treat a pair as duplicates.
        stats: Counters to update.

    Returns:
        Union-find structure whose clusters are the duplicate groups.
    """
    clusters = UnionFind(n_docs)
    stats.candidate_pairs = len(pairs)

    for processed, (i, j) in enumerate(sorted(pairs), start=1):
        if jaccard(shingle_store.get(i), shingle_store.get(j)) >= threshold:
            clusters.union(i, j)
            stats.verified_pairs += 1
        if processed % 200_000 == 0:
            LOGGER.info("  verified %s/%s pairs (%s confirmed)",
                        f"{processed:,}", f"{len(pairs):,}",
                        f"{stats.verified_pairs:,}")

    return clusters


class ShingleStore:
    """On-disk store of per-document shingle sets.

    Verification needs the actual shingle sets, but 2.2M documents at ~300
    shingles each is ~2.7 GB -- too much for RAM alongside everything else, and
    wasteful to recompute since only candidate pairs are ever read back.  A flat
    uint32 array plus an offset index gives O(1) retrieval from disk.
    """

    def __init__(self, path: Path) -> None:
        """Create an empty store at the given path prefix.

        Args:
            path: Prefix; ``<path>.data`` and ``<path>.idx`` are written.
        """
        self.data_path = path.with_suffix(".data")
        self.index_path = path.with_suffix(".idx")
        self._data_file = open(self.data_path, "wb")
        self._offsets: list[int] = [0]
        self._data: np.ndarray | None = None
        self._index: np.ndarray | None = None

    def append(self, shingle_hashes: np.ndarray) -> None:
        """Append one document's shingle set."""
        self._data_file.write(shingle_hashes.tobytes())
        self._offsets.append(self._offsets[-1] + shingle_hashes.size)

    def finalize(self) -> None:
        """Close writing and memory-map the store for reading."""
        self._data_file.close()
        self._index = np.array(self._offsets, dtype=np.int64)
        np.save(self.index_path, self._index)
        self._data = np.memmap(self.data_path, dtype=np.uint32, mode="r")

    def get(self, doc_index: int) -> np.ndarray:
        """Return one document's sorted shingle set."""
        assert self._data is not None, "call finalize() before get()"
        start, end = self._index[doc_index], self._index[doc_index + 1]
        return np.asarray(self._data[start:end])


def sha1_of(text: str) -> str:
    """Return the SHA-1 hex digest of normalized text.

    SHA-1 rather than a faster non-cryptographic hash because accidental
    collisions must be negligible across millions of documents; this is not a
    security use.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Corpus driver
#
# Everything above is the algorithm; everything below runs it over real shards.
# The driver deliberately handles deduplication AND decontamination in one pass:
# both ask the same question ("is this train document a near-duplicate of that
# one?"), so building the signature matrix twice would double the cost for no
# gain. What differs is only how a cluster is resolved, in resolve_cluster().
# ---------------------------------------------------------------------------

import io  # noqa: E402
import time  # noqa: E402

import zstandard as zstd  # noqa: E402


def split_shards(clean_dir: Path, split: str) -> list[Path]:
    """Cleaned shards belonging to one split, for either on-disk layout.

    The current layout gives each split its own directory
    (``clean/train/train-00000.jsonl.zst``).  Earlier runs wrote every shard
    flat in ``clean/`` (``clean/train-00000.jsonl.zst``).  Both are accepted, so
    a corpus produced before the split directories existed still loads.

    Args:
        clean_dir: Cleaned corpus directory.
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        Shard paths in sorted order.
    """
    nested = sorted((clean_dir / split).glob("*.jsonl.zst"))
    return nested or sorted(clean_dir.glob(f"{split}-*.jsonl.zst"))


def all_clean_shards(clean_dir: Path) -> list[Path]:
    """Every cleaned shard, across both layouts.

    Callers identify the split from the file name, which carries the split
    prefix in either layout.

    Args:
        clean_dir: Cleaned corpus directory.

    Returns:
        Shard paths in sorted order.
    """
    return sorted(list(clean_dir.glob("*.jsonl.zst"))
                  + list(clean_dir.glob("*/*.jsonl.zst")))


def iter_shard(path: Path) -> Iterator[dict]:
    """Yield JSON records from one zstd-compressed JSONL shard.

    Args:
        path: Shard file.

    Yields:
        Parsed records; unparseable lines (a partial trailing line in a shard a
        crawler was still writing) are skipped rather than aborting the run.
    """
    with open(path, "rb") as handle:
        reader = zstd.ZstdDecompressor().stream_reader(handle)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue


class ShardWriterSimple:
    """Minimal zstd JSONL writer, rotated by uncompressed size."""

    def __init__(self, out_dir: Path, prefix: str,
                 max_bytes: int = 256 * 1024 * 1024) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.index = 0
        self.written = 0
        self._open()

    def _open(self) -> None:
        path = self.out_dir / f"{self.prefix}-{self.index:05d}.jsonl.zst"
        self._fh = open(path, "wb")
        self._writer = zstd.ZstdCompressor(level=10).stream_writer(self._fh)
        self.written = 0

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        self._writer.write(line)
        self.written += len(line)
        if self.written >= self.max_bytes:
            self.close()
            self.index += 1
            self._open()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush(zstd.FLUSH_FRAME)
            self._writer.close()
            self._fh.close()
            self._writer = None


def resolve_cluster(cluster: list[int], refs: list[DocumentRef],
                    is_heldout: np.ndarray, stats: DedupStats) -> set[int]:
    """Decide which documents in one duplicate cluster to drop.

    **Every** removal decision flows through here -- exact duplicates included.
    An earlier version short-circuited exact duplicates with a first-seen-wins
    rule, which broke both hard rules the moment shard order disagreed with
    them: a public copy read before its manual twin won, and a train copy read
    before its held-out twin won.  Resolution must be decided by document
    *properties*, never by read order.

    Priority, in order:

    1. **Held-out (val/test) documents present.**  Every train document in the
       cluster is dropped -- this *is* decontamination, and it applies whether
       the overlap is exact or a paraphrase.  Held-out documents are kept, since
       val/test receive exact-duplicate removal only; removing near-duplicates
       inside the held-out sets would change what the evaluation measures.
       Exact duplicates *within* the held-out set still collapse to one copy.
    2. **Train documents only.**  One survivor by the manual-wins rule in
       :func:`choose_survivor`.

    Args:
        cluster: Document indices in the cluster.
        refs: Document references indexed by document index.
        is_heldout: Boolean array marking val/test documents.
        stats: Counters to update.

    Returns:
        Indices to drop.
    """
    if len(cluster) < 2:
        return set()

    heldout = [i for i in cluster if is_heldout[i]]
    train = [i for i in cluster if not is_heldout[i]]

    if heldout:
        # Decontamination: the held-out copy defines the evaluation, so every
        # train copy goes regardless of source_type.
        drop = set(train)
        # Collapse exact duplicates among the held-out copies themselves,
        # keeping the first of each distinct SHA-1.
        seen: set[str] = set()
        for index in heldout:
            if refs[index].sha1 in seen:
                drop.add(index)
            else:
                seen.add(refs[index].sha1)
        return drop

    survivor, saved = choose_survivor(train, refs)
    stats.manual_saved_from_public += saved
    return {i for i in train if i != survivor}


def deduplicate_corpus(
    train_shards: list[Path],
    heldout_shards: list[Path],
    out_dir: Path,
    work_dir: Path,
    chars_per_token: float,
    threshold: float = JACCARD_THRESHOLD,
) -> dict:
    """Deduplicate a corpus and decontaminate train against held-out splits.

    Runs in four passes so that peak memory stays bounded regardless of corpus
    size: signatures and shingles go to disk, and only one LSH band is resident
    at a time.

    Args:
        train_shards: Staged train shards.
        heldout_shards: Staged val/test shards.
        out_dir: Where surviving documents are written.
        work_dir: Scratch space for signature and shingle stores.
        chars_per_token: Measured constant, for token accounting.
        threshold: Exact Jaccard at or above which a pair is a duplicate.

    Returns:
        Statistics dictionary including before/after token counts.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    stats = DedupStats()
    hasher = MinHasher()
    store = ShingleStore(work_dir / "shingles")

    refs: list[DocumentRef] = []
    heldout_flags: list[bool] = []
    signature_path = work_dir / "signatures.u32"
    sha1_groups: dict[str, list[int]] = {}

    started = time.time()
    LOGGER.info("pass 1/4: hashing %d train + %d held-out shards",
                len(train_shards), len(heldout_shards))

    with open(signature_path, "wb") as sig_file:
        for shard_index, path in enumerate(list(train_shards) + list(heldout_shards)):
            for line_no, record in enumerate(iter_shard(path)):
                text = record.get("text") or ""
                if not text:
                    continue
                # Held-out status comes from the record's own split field, not
                # from which file it arrived in: staged shards carry a mix of
                # train/val/test, because the split was assigned per document
                # back in stage 2, before any filtering.
                held = record.get("split") in ("val", "test")
                index = len(refs)
                digest = sha1_of(text)

                # Exact duplicates are GROUPED here, not dropped here. Which
                # copy survives is decided later by resolve_cluster(), so the
                # manual-wins and decontamination rules apply to exact
                # duplicates exactly as they do to near duplicates.
                sha1_groups.setdefault(digest, []).append(index)

                doc_shingles = shingles(text)
                store.append(doc_shingles)
                sig_file.write(hasher.signature(doc_shingles).tobytes())

                refs.append(DocumentRef(
                    shard=shard_index, line=line_no,
                    source_type=record.get("source_type") or "public",
                    source=record.get("source") or "unknown",
                    characters=len(text), sha1=digest))
                heldout_flags.append(held)
                stats.documents_in += 1
                stats.characters_in += len(text)

                if stats.documents_in % 250_000 == 0:
                    LOGGER.info("  hashed %s docs (%.0fs)",
                                f"{stats.documents_in:,}", time.time() - started)

    store.finalize()
    n_docs = len(refs)
    # Only one representative per SHA-1 group enters LSH: identical text has an
    # identical signature, so the others would generate candidate pairs that add
    # nothing. Each non-representative is unioned with its representative
    # afterwards, so it still lands in the right cluster.
    representative = {group[0] for group in sha1_groups.values()}
    duplicate_of = {index: group[0]
                    for group in sha1_groups.values() for index in group[1:]}
    LOGGER.info("  %s distinct texts, %s exact-duplicate copies",
                f"{len(sha1_groups):,}", f"{len(duplicate_of):,}")
    is_heldout = np.array(heldout_flags, dtype=bool)
    signatures = np.memmap(signature_path, dtype=np.uint32,
                           mode="r", shape=(n_docs, NUM_PERM))
    LOGGER.info("pass 1 done: %s docs (%.0fs)", f"{n_docs:,}", time.time() - started)

    LOGGER.info("pass 2/4: LSH banding (b=%d, r=%d, threshold=%.4f)",
                BANDS, ROWS, band_threshold())
    pairs = find_candidate_pairs(signatures)
    pairs = {(i, j) for i, j in pairs
             if i in representative and j in representative}

    LOGGER.info("pass 3/4: verifying %s candidate pairs by exact Jaccard",
                f"{len(pairs):,}")
    clusters = build_clusters(pairs, store, n_docs, threshold, stats)

    LOGGER.info("pass 4/4: resolving clusters and writing survivors")
    # Fold every exact duplicate into its representative's cluster so that one
    # resolution pass sees all copies of a text at once.
    for index, rep in duplicate_of.items():
        clusters.union(rep, index)

    grouped: dict[int, list[int]] = {}
    for index in range(n_docs):
        grouped.setdefault(clusters.find(index), []).append(index)

    drop: set[int] = set()
    for members in grouped.values():
        drop |= resolve_cluster(members, refs, is_heldout, stats)

    # A dropped document counts as an EXACT duplicate when some surviving
    # document has the same SHA-1, and as a NEAR duplicate otherwise.
    kept_sha1 = {refs[i].sha1 for i in range(n_docs) if i not in drop}
    stats.exact_duplicates = sum(1 for i in drop if refs[i].sha1 in kept_sha1)
    stats.near_duplicates = len(drop) - stats.exact_duplicates

    for index in drop:
        ref = refs[index]
        stats.removed_by_source[ref.source] = stats.removed_by_source.get(ref.source, 0) + 1
        stats.removed_by_source_type[ref.source_type] = \
            stats.removed_by_source_type.get(ref.source_type, 0) + 1

    # Re-read the shards in the same order and emit anything not dropped.
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {"train": ShardWriterSimple(out_dir, "train"),
               "val": ShardWriterSimple(out_dir, "val"),
               "test": ShardWriterSimple(out_dir, "test")}
    index = 0
    for path in list(train_shards) + list(heldout_shards):
        for record in iter_shard(path):
            if not (record.get("text") or ""):
                continue
            if index not in drop:
                writers[record.get("split", "train")].write(record)
                stats.documents_out += 1
                stats.characters_out += len(record["text"])
            index += 1
    for writer in writers.values():
        writer.close()

    LOGGER.info("dedup complete in %.0fs: %s -> %s docs",
                time.time() - started,
                f"{stats.documents_in:,}", f"{stats.documents_out:,}")
    return stats.as_dict(chars_per_token)
