"""Resumable URL frontier backed by SQLite.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

The manual-collection crawl runs for roughly nine hours per language, so it must
survive being interrupted: a crash or a Ctrl-C should cost the last few in-flight
requests, not the whole run.  This module holds the crawl state on disk so the
fetcher itself can stay stateless.

SQLite rather than an in-memory set because this machine has ~14 GB of RAM with
only ~4 GB free.  Holding a Python ``set`` of ~1.5M URL strings plus per-URL
metadata would cost well over a gigabyte of that budget, and it would be lost on
every restart.  SQLite keeps memory flat and gives crash-safe deduplication for
free via a primary key.

URLs are keyed by the SHA-1 of the URL rather than the URL text so the primary
key is fixed-width, which keeps the index small across millions of rows.

This module stores URLs and crawl bookkeeping only -- never corpus text, and
never anything language-specific.  Each language gets its own database file, so
the two pipelines share code but no state.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

# Frontier states.  Kept as small ints so the status index stays compact.
PENDING = 0
IN_FLIGHT = 1
DONE = 2
FAILED = 3
SKIPPED = 4

_STATUS_NAMES = {
    PENDING: "pending",
    IN_FLIGHT: "in_flight",
    DONE: "done",
    FAILED: "failed",
    SKIPPED: "skipped",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
    url_hash   BLOB PRIMARY KEY,   -- sha1(url), 20 bytes, fixed width
    url        TEXT NOT NULL,
    host       TEXT NOT NULL,
    source     TEXT NOT NULL,      -- site name from the sources.json inventory
    state      INTEGER NOT NULL,
    http_status INTEGER,
    n_chars    INTEGER,            -- extracted text length, for yield tracking
    attempts   INTEGER NOT NULL DEFAULT 0,
    updated_at REAL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_state      ON urls(state);
CREATE INDEX IF NOT EXISTS idx_host_state ON urls(host, state);
"""


def url_key(url: str) -> bytes:
    """Return the 20-byte SHA-1 key for a URL.

    Args:
        url: Absolute URL.

    Returns:
        Raw SHA-1 digest used as the frontier primary key.
    """
    return hashlib.sha1(url.encode("utf-8")).digest()


@dataclass
class ClaimedUrl:
    """A URL handed to a fetcher worker."""

    key: bytes
    url: str
    host: str
    source: str


class UrlFrontier:
    """Crash-safe store of discovered URLs and their crawl state.

    Typical lifecycle::

        frontier = UrlFrontier(Path("artifacts/corpus/frontier.db"))
        frontier.add_many(urls, host="www.jagran.com", source="Dainik Jagran")
        for item in frontier.claim_batch(200):
            ...  # fetch
            frontier.mark_done(item.key, 200, n_chars=4200)
        frontier.close()
    """

    def __init__(self, db_path: Path | str, reset_in_flight: bool = True) -> None:
        """Open (creating if needed) the frontier database.

        Args:
            db_path: Path to the SQLite file.  Parent directories are created.
            reset_in_flight: Return URLs left in the ``IN_FLIGHT`` state by a
                previous crashed run to ``PENDING``.  This is what makes a
                restart pick up exactly where the crash left off, and should
                only be disabled if several fetcher processes share one database.
        """
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60.0, isolation_level=None)
        # WAL keeps readers (progress monitoring) from blocking the writer, and
        # NORMAL sync is the right trade for a re-runnable crawl: a power loss
        # can lose the last transaction, which we would simply re-fetch.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        if reset_in_flight:
            self.recover_in_flight()

    def recover_in_flight(self) -> int:
        """Move any ``IN_FLIGHT`` URLs back to ``PENDING`` after a crash.

        Returns:
            Number of URLs recovered.
        """
        cur = self.conn.execute(
            "UPDATE urls SET state=?, updated_at=? WHERE state=?",
            (PENDING, time.time(), IN_FLIGHT),
        )
        return cur.rowcount

    def add_many(self, urls: list[str], host: str, source: str) -> int:
        """Insert newly discovered URLs, ignoring ones already known.

        Deduplication is done by the primary key, so calling this repeatedly with
        overlapping URL sets is safe and cheap -- which matters because sitemap
        re-enumeration during a resumed run will rediscover most of the same URLs.

        Args:
            urls: Absolute URLs to add.
            host: Hostname these URLs belong to, used for per-host rate limiting.
            source: Human-readable site name from the ``sources.json`` inventory,
                carried through to the corpus as provenance.

        Returns:
            Count of URLs actually inserted (i.e. not previously known).
        """
        if not urls:
            return 0
        now = time.time()
        rows = [(url_key(u), u, host, source, PENDING, now) for u in urls]
        cur = self.conn.execute("BEGIN")
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO urls"
            " (url_hash, url, host, source, state, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            rows,
        )
        inserted = cur.rowcount
        self.conn.execute("COMMIT")
        return inserted

    def claim_batch(self, limit: int, host: str | None = None) -> list[ClaimedUrl]:
        """Atomically claim up to ``limit`` pending URLs for fetching.

        Claiming flips the rows to ``IN_FLIGHT`` inside one transaction, so a
        crash leaves them recoverable rather than silently lost or double-fetched.

        Args:
            limit: Maximum number of URLs to claim.
            host: Restrict the claim to a single host.  Used to feed per-host
                worker queues so that rate limiting stays per-host.

        Returns:
            Claimed URLs, possibly fewer than ``limit`` (empty when the frontier
            is drained).
        """
        if host is None:
            sql = "SELECT url_hash, url, host, source FROM urls WHERE state=? LIMIT ?"
            params: tuple = (PENDING, limit)
        else:
            sql = (
                "SELECT url_hash, url, host, source FROM urls"
                " WHERE state=? AND host=? LIMIT ?"
            )
            params = (PENDING, host, limit)

        self.conn.execute("BEGIN IMMEDIATE")
        rows = self.conn.execute(sql, params).fetchall()
        if rows:
            self.conn.executemany(
                "UPDATE urls SET state=?, attempts=attempts+1, updated_at=?"
                " WHERE url_hash=?",
                [(IN_FLIGHT, time.time(), r[0]) for r in rows],
            )
        self.conn.execute("COMMIT")
        return [ClaimedUrl(key=r[0], url=r[1], host=r[2], source=r[3]) for r in rows]

    def mark(
        self,
        key: bytes,
        state: int,
        http_status: int | None = None,
        n_chars: int | None = None,
    ) -> None:
        """Record the outcome of one fetch.

        Args:
            key: Frontier key returned by :meth:`claim_batch`.
            state: One of ``DONE``, ``FAILED`` or ``SKIPPED``.
            http_status: HTTP status observed, if any.
            n_chars: Characters of text extracted, used to compute the
                HTML-to-text yield reported in the dataset statistics.
        """
        self.conn.execute(
            "UPDATE urls SET state=?, http_status=?, n_chars=?, updated_at=?"
            " WHERE url_hash=?",
            (state, http_status, n_chars, time.time(), key),
        )

    def requeue(self, key: bytes, max_attempts: int = 3) -> bool:
        """Return a transiently failed URL to ``PENDING`` unless it is exhausted.

        Args:
            key: Frontier key.
            max_attempts: Attempt ceiling before the URL is marked ``FAILED``.

        Returns:
            ``True`` if the URL was requeued, ``False`` if it was retired as
            permanently failed.
        """
        row = self.conn.execute(
            "SELECT attempts FROM urls WHERE url_hash=?", (key,)
        ).fetchone()
        attempts = row[0] if row else max_attempts
        if attempts >= max_attempts:
            self.mark(key, FAILED)
            return False
        self.conn.execute(
            "UPDATE urls SET state=?, updated_at=? WHERE url_hash=?",
            (PENDING, time.time(), key),
        )
        return True

    def counts(self) -> dict[str, int]:
        """Return the number of URLs in each state.

        Returns:
            Mapping of state name to count, e.g. ``{"pending": 12000, "done": 340}``.
        """
        rows = self.conn.execute(
            "SELECT state, COUNT(*) FROM urls GROUP BY state"
        ).fetchall()
        return {_STATUS_NAMES.get(s, str(s)): n for s, n in rows}

    def per_host_progress(self) -> list[tuple[str, int, int, int]]:
        """Summarize progress per host, for monitoring a long crawl.

        Returns:
            Rows of ``(host, done, pending, extracted_chars)`` sorted by host.
        """
        return self.conn.execute(
            "SELECT host,"
            "       SUM(state=2),"
            "       SUM(state=0),"
            "       COALESCE(SUM(n_chars), 0)"
            " FROM urls GROUP BY host ORDER BY host"
        ).fetchall()

    def hosts_with_pending(self) -> list[str]:
        """Return hosts that still have pending URLs.

        Returns:
            Hostnames, used to decide which per-host workers to run.
        """
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT host FROM urls WHERE state=?", (PENDING,)
            )
        ]

    def close(self) -> None:
        """Flush the write-ahead log and close the database."""
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            self.conn.close()

    def __enter__(self) -> "UrlFrontier":
        """Support use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the database on context exit."""
        self.close()
