"""Polite asynchronous crawler for manual corpus collection.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

This module supplies the >=20% manually collected portion of each corpus.  Two
stages, used independently:

1. :class:`SitemapEnumerator` -- expand sitemap indexes into article URLs and
   load them into the :mod:`data.pipeline.frontier` database.
2. :class:`PoliteCrawler` -- fetch those URLs, extract article text, and append
   it to compressed JSONL shards.

Two decisions worth stating, because both are load-bearing:

**Extraction happens inline, not in a later pass.**  A news page is ~150 KB of
HTML whose article body is ~6 KB of Devanagari -- roughly a 4% yield.  Storing
raw HTML for the full crawl would cost ~40 GB per language to keep ~1.6 GB of
text.  Instead trafilatura runs in the fetch loop and only text is persisted.  A
configurable random sample of raw HTML (default 5%) *is* kept, so extraction
bugs can be diagnosed and the extraction step re-run on a sample without
re-crawling.

**Rate limiting is per host, not global.**  Politeness is a per-server property.
One shared limiter would either hammer a small site or waste the whole crawl
budget waiting on the slowest one.  Each host gets its own worker honouring its
own delay, and hosts run concurrently, which is what makes ~8 pages/second
aggregate achievable while staying at 1 request/second per site.

``robots.txt`` is fetched and honoured for every host, including any
``Crawl-delay`` directive, which overrides the configured rate when it is
stricter.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import logging
import random
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import urljoin, urlparse

import aiohttp
import zstandard as zstd
from lxml import etree

from data.pipeline.frontier import DONE, FAILED, SKIPPED, ClaimedUrl, UrlFrontier

LOGGER = logging.getLogger(__name__)

#: Sitemap XML namespace, as emitted by every sitemap we probed.
_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

#: URL suffixes that never contain article prose.
_SKIP_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".mp3", ".m3u8", ".pdf", ".zip", ".css", ".js",
)


@dataclass
class CrawlConfig:
    """Crawl politeness and output settings.

    Attributes:
        user_agent: Sent on every request and matched against ``robots.txt``.
        requests_per_second_per_host: Steady-state rate for a single host.  A
            stricter ``Crawl-delay`` in ``robots.txt`` takes precedence.
        max_concurrent_hosts: How many per-host workers run at once.  With the
            default 1 req/s this is also the approximate aggregate page rate.
        max_in_flight_per_host: Requests allowed outstanding to one host at
            once.  Politeness is a constraint on request *rate*, not on
            concurrency depth: waiting for each response before issuing the next
            makes throughput hostage to server latency, which on slow hosts
            (~2.7 s/response) cost roughly two thirds of the configured rate.  Requests are issued on a fixed schedule and this caps how many
            may be outstanding, so a stalled server cannot cause a pile-up.
        timeout_seconds: Total per-request timeout.
        max_retries: Attempts before a URL is retired as failed.
        raw_html_sample_rate: Fraction of pages whose raw HTML is retained for
            debugging and for re-running extraction without re-crawling.
        min_chars: Extracted documents shorter than this are discarded as
            navigation chrome or stubs rather than written out.
        shard_max_bytes: Approximate uncompressed size at which the output shard
            is rotated.
        respect_robots_txt: Kept configurable only so the value can be logged
            with the run config; it must stay ``True``.
    """

    user_agent: str = (
        "Mozilla/5.0 (compatible; LMA-course-research-crawler/1.0; "
        "academic monolingual corpus construction)"
    )
    requests_per_second_per_host: float = 1.0
    max_concurrent_hosts: int = 8
    max_in_flight_per_host: int = 4
    timeout_seconds: int = 30
    max_retries: int = 3
    raw_html_sample_rate: float = 0.05
    min_chars: int = 200
    shard_max_bytes: int = 256 * 1024 * 1024
    respect_robots_txt: bool = True


@dataclass
class CrawlStats:
    """Running totals for one crawl session."""

    fetched: int = 0
    extracted: int = 0
    too_short: int = 0
    http_errors: int = 0
    exceptions: int = 0
    robots_skipped: int = 0
    chars_written: int = 0
    html_bytes: int = 0
    started_at: float = field(default_factory=time.time)

    def yield_ratio(self) -> float:
        """Fraction of downloaded HTML bytes that survived as text.

        This is the empirical version of the ~4% extraction yield the corpus
        sizing math assumes, and is reported in the dataset statistics.

        Returns:
            Ratio of extracted UTF-8 text bytes to downloaded HTML bytes, or
            ``0.0`` before any bytes have been downloaded.
        """
        return (self.chars_written * 2.68) / self.html_bytes if self.html_bytes else 0.0

    def summary(self) -> str:
        """Return a one-line human-readable progress summary."""
        elapsed = max(time.time() - self.started_at, 1e-6)
        return (
            f"fetched={self.fetched} extracted={self.extracted} "
            f"short={self.too_short} http_err={self.http_errors} "
            f"exc={self.exceptions} robots_skip={self.robots_skipped} "
            f"chars={self.chars_written:,} "
            f"rate={self.fetched / elapsed:.1f}/s "
            f"yield={self.yield_ratio():.1%}"
        )


class ShardWriter:
    """Append-only writer producing zstd-compressed JSONL shards.

    Zstandard at level 10 gives roughly 4x on Devanagari JSONL at a throughput
    far above the crawl rate, so compression is never the bottleneck.  Shards are
    rotated by size so that a corrupted tail costs one shard rather than the
    whole crawl, and so downstream cleaning can process them in parallel.
    """

    def __init__(self, out_dir: Path, prefix: str, max_bytes: int) -> None:
        """Prepare the output directory and open the first shard.

        Args:
            out_dir: Directory to write shards into; created if absent.
            prefix: Shard filename prefix, e.g. ``"manual"``.
            max_bytes: Approximate uncompressed byte count per shard.
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_bytes = max_bytes
        self._index = self._next_shard_index()
        self._written = 0
        self._fh: io.BufferedWriter | None = None
        self._writer = None
        self._open()

    def _next_shard_index(self) -> int:
        """Return the next unused shard number, so restarts never overwrite.

        Returns:
            One past the highest existing shard index for this prefix.
        """
        existing = sorted(self.out_dir.glob(f"{self.prefix}-*.jsonl.zst"))
        if not existing:
            return 0
        last = existing[-1].name.split("-")[-1].split(".")[0]
        return int(last) + 1

    def _open(self) -> None:
        """Open a new shard file and its zstd stream writer."""
        path = self.out_dir / f"{self.prefix}-{self._index:05d}.jsonl.zst"
        self._fh = open(path, "wb")
        self._writer = zstd.ZstdCompressor(level=10).stream_writer(self._fh)
        self._written = 0
        LOGGER.info("opened shard %s", path.name)

    def write(self, record: dict) -> int:
        """Append one JSON record, rotating the shard when it grows too large.

        Args:
            record: JSON-serializable document record.

        Returns:
            Number of uncompressed bytes written.
        """
        line = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        self._writer.write(line)
        self._written += len(line)
        if self._written >= self.max_bytes:
            self.close()
            self._index += 1
            self._open()
        return len(line)

    def close(self) -> None:
        """Flush and close the current shard."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class RobotsPolicy:
    """Per-host ``robots.txt`` cache and access check.

    Fetched once per host and reused.  A host whose ``robots.txt`` cannot be
    retrieved is treated as *allowed*, matching the convention that an
    unreachable robots file is not a prohibition -- but a host that returns a
    parseable file disallowing our user agent is skipped entirely.
    """

    def __init__(self, user_agent: str) -> None:
        """Initialize an empty cache.

        Args:
            user_agent: Agent string to evaluate rules against.
        """
        self.user_agent = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._delays: dict[str, float | None] = {}

    async def load(self, session: aiohttp.ClientSession, host: str) -> None:
        """Fetch and parse ``robots.txt`` for a host if not already cached.

        Args:
            session: Shared HTTP session.
            host: Hostname to load rules for.
        """
        if host in self._parsers:
            return
        parser: urllib.robotparser.RobotFileParser | None = None
        try:
            async with session.get(f"https://{host}/robots.txt") as resp:
                if resp.status == 200:
                    text = await resp.text(errors="ignore")
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse(text.splitlines())
        except Exception as exc:  # network failure -> treat as unrestricted
            LOGGER.warning("robots.txt fetch failed for %s: %s", host, exc)
        self._parsers[host] = parser
        delay = None
        if parser is not None:
            try:
                delay = parser.crawl_delay(self.user_agent)
            except Exception:
                delay = None
        self._delays[host] = float(delay) if delay else None

    def allowed(self, host: str, url: str) -> bool:
        """Return whether ``url`` may be fetched under the host's rules.

        Args:
            host: Hostname owning the rules.
            url: Absolute URL to check.

        Returns:
            ``True`` when allowed or when no rules are known.
        """
        parser = self._parsers.get(host)
        if parser is None:
            return True
        try:
            return parser.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def crawl_delay(self, host: str) -> float | None:
        """Return the host's declared ``Crawl-delay`` in seconds, if any.

        Args:
            host: Hostname.

        Returns:
            Delay in seconds, or ``None`` when unspecified.
        """
        return self._delays.get(host)


class SitemapEnumerator:
    """Expand sitemap indexes into a flat list of candidate article URLs.

    Handles the three shapes seen across the surveyed sites: a ``<sitemapindex>``
    pointing at further sitemaps (recursed into, depth-limited), a flat
    ``<urlset>``, and gzip-compressed variants of either.
    """

    def __init__(self, session: aiohttp.ClientSession, max_depth: int = 3) -> None:
        """Store the HTTP session and recursion limit.

        Args:
            session: Shared HTTP session.
            max_depth: Maximum nesting of sitemap indexes to follow.  Real sites
                nest at most two levels; the limit guards against cycles.
        """
        self.session = session
        self.max_depth = max_depth
        self._seen: set[str] = set()

    async def _fetch_xml(self, url: str) -> bytes | None:
        """Download a sitemap, transparently decompressing gzip.

        Args:
            url: Sitemap URL.

        Returns:
            Raw XML bytes, or ``None`` if unavailable.
        """
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    LOGGER.warning("sitemap %s -> HTTP %s", url, resp.status)
                    return None
                raw = await resp.read()
        except Exception as exc:
            LOGGER.warning("sitemap %s failed: %s", url, exc)
            return None
        # Some hosts serve .xml.gz without a Content-Encoding header.
        if raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
            except Exception:
                return None
        return raw

    async def enumerate(
        self,
        sitemap_urls: Sequence[str],
        url_filter: str | None = None,
        on_urls: "Callable[[list[str]], None] | None" = None,
    ) -> list[str]:
        """Recursively expand sitemaps into article URLs.

        Args:
            sitemap_urls: Entry-point sitemap URLs.
            url_filter: Optional substring every returned URL must contain.  Used
                for multi-language hosts -- BBC serves many languages from one
                sitemap, so ``"/hindi/"`` keeps only Hindi URLs at discovery
                time rather than relying on language ID to clean up afterwards.
            on_urls: Optional callback invoked with each child sitemap's URLs as
                soon as they are parsed.  Large archives fan out into hundreds of
                child sitemaps and can take an hour to walk; committing
                incrementally means an interrupted run keeps the URLs it already
                found, and lets the crawl start before discovery finishes.

        Returns:
            Deduplicated candidate URLs.
        """
        out: list[str] = []
        await self._walk(list(sitemap_urls), 0, url_filter, out, on_urls)
        return out

    async def _walk(
        self,
        urls: list[str],
        depth: int,
        url_filter: str | None,
        out: list[str],
        on_urls: "Callable[[list[str]], None] | None" = None,
    ) -> None:
        """Depth-first sitemap expansion, appending page URLs to ``out``.

        Args:
            urls: Sitemap URLs at this level.
            depth: Current recursion depth.
            url_filter: Optional substring filter for page URLs.
            out: Accumulator for discovered page URLs.
            on_urls: Optional per-sitemap commit callback.
        """
        if depth > self.max_depth:
            return
        for sm_url in urls:
            if sm_url in self._seen:
                continue
            self._seen.add(sm_url)

            raw = await self._fetch_xml(sm_url)
            if raw is None:
                continue
            try:
                root = etree.fromstring(raw)
            except etree.XMLSyntaxError as exc:
                LOGGER.warning("sitemap %s unparseable: %s", sm_url, exc)
                continue

            tag = etree.QName(root).localname if root.tag else ""
            locs = [
                (e.text or "").strip()
                for e in root.iterfind(".//sm:loc", _SITEMAP_NS)
            ]
            if not locs:  # some hosts omit the namespace declaration
                locs = [(e.text or "").strip() for e in root.iterfind(".//loc")]
            locs = [urljoin(sm_url, loc) for loc in locs if loc]

            if tag == "sitemapindex":
                LOGGER.info("sitemap index %s -> %d child sitemaps", sm_url, len(locs))
                await self._walk(locs, depth + 1, url_filter, out, on_urls)
            else:
                kept = [u for u in locs if _is_candidate_page(u, url_filter)]
                LOGGER.info("sitemap %s -> %d/%d page URLs", sm_url, len(kept), len(locs))
                out.extend(kept)
                if on_urls and kept:
                    on_urls(kept)


def _is_candidate_page(url: str, url_filter: str | None) -> bool:
    """Return whether a URL plausibly points at an article page.

    Args:
        url: Absolute URL from a sitemap.
        url_filter: Optional substring the URL must contain.

    Returns:
        ``True`` when the URL should be enqueued for fetching.
    """
    if not url.startswith(("http://", "https://")):
        return False
    if url.lower().endswith(_SKIP_SUFFIXES):
        return False
    if url_filter and url_filter not in url:
        return False
    return True


def extract_text(html: str, url: str) -> tuple[str, str]:
    """Extract article body text and title from an HTML page.

    Runs trafilatura with comments and tables excluded: comment sections are
    frequently romanized or in another language entirely, and would pollute a
    monolingual corpus that the language filter might not catch.

    This function is CPU-bound and is therefore invoked through a thread
    executor by the crawler, so it never blocks the event loop.

    Args:
        html: Raw page HTML.
        url: Source URL, passed to trafilatura for relative-link resolution.

    Returns:
        Tuple of ``(text, title)``; ``("", "")`` when nothing was extracted.
    """
    # Imported lazily so that importing this module stays cheap for the
    # enumeration-only entry points.
    import trafilatura

    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
    except Exception as exc:
        LOGGER.debug("extraction failed for %s: %s", url, exc)
        return "", ""
    if not text:
        return "", ""

    title = ""
    try:
        meta = trafilatura.extract_metadata(html)
        if meta is not None and meta.title:
            title = meta.title
    except Exception:
        pass
    return text, title


class PoliteCrawler:
    """Fetch frontier URLs at a per-host rate limit and write extracted text.

    Args:
        frontier: URL frontier to draw work from and report outcomes to.
        writer: Destination for extracted documents.
        config: Politeness and output settings.
        source_type: Provenance tag written on every record.  Always
            ``"manual"`` here -- this crawler *is* the manual collection, and the
            tag is what the >=20% audit counts.
        raw_html_dir: Directory for the sampled raw HTML, if sampling is enabled.
    """

    def __init__(
        self,
        frontier: UrlFrontier,
        writer: ShardWriter,
        config: CrawlConfig,
        source_type: str = "manual",
        raw_html_dir: Path | None = None,
    ) -> None:
        """Initialize the crawler and its shared state."""
        self.frontier = frontier
        self.writer = writer
        self.config = config
        self.source_type = source_type
        self.robots = RobotsPolicy(config.user_agent)
        self.stats = CrawlStats()
        self.raw_html_dir = raw_html_dir
        if raw_html_dir is not None:
            Path(raw_html_dir).mkdir(parents=True, exist_ok=True)
        self._write_lock = asyncio.Lock()
        self._rng = random.Random(20260813)

    async def run(self, max_pages: int | None = None) -> CrawlStats:
        """Crawl until the frontier is drained or ``max_pages`` is reached.

        One worker is started per host, up to ``max_concurrent_hosts``, so hosts
        are fetched in parallel while each stays within its own rate limit.

        Args:
            max_pages: Optional ceiling on pages fetched this session, used for
                smoke tests and for topping the corpus up by a known amount.

        Returns:
            Statistics for the session.
        """
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_hosts * 2, ssl=False
        )
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept-Language": "hi,ne,en;q=0.5",
        }
        budget = asyncio.Semaphore(self.config.max_concurrent_hosts)

        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector, headers=headers
        ) as session:
            hosts = self.frontier.hosts_with_pending()
            LOGGER.info("crawling %d hosts", len(hosts))
            for host in hosts:
                await self.robots.load(session, host)

            self._remaining = max_pages
            await asyncio.gather(
                *[self._host_worker(session, h, budget) for h in hosts],
                return_exceptions=True,
            )
        self.writer.close()
        return self.stats

    async def _host_worker(
        self, session: aiohttp.ClientSession, host: str, budget: asyncio.Semaphore
    ) -> None:
        """Drain one host's pending URLs at that host's permitted rate.

        Args:
            session: Shared HTTP session.
            host: Hostname this worker is responsible for.
            budget: Semaphore capping how many hosts run concurrently.
        """
        async with budget:
            declared = self.robots.crawl_delay(host)
            interval = max(
                1.0 / max(self.config.requests_per_second_per_host, 1e-9),
                declared or 0.0,
            )
            if declared:
                LOGGER.info("%s declares Crawl-delay=%.1fs", host, declared)

            in_flight = asyncio.Semaphore(self.config.max_in_flight_per_host)
            pending: set[asyncio.Task] = set()

            async def issue(item: ClaimedUrl) -> None:
                """Fetch one URL and release its in-flight slot.

                Args:
                    item: Claimed frontier entry.
                """
                try:
                    await self._fetch_one(session, item)
                finally:
                    in_flight.release()

            try:
                while True:
                    if self._remaining is not None and self._remaining <= 0:
                        return
                    batch = self.frontier.claim_batch(50, host=host)
                    if not batch:
                        return
                    for item in batch:
                        if self._remaining is not None:
                            if self._remaining <= 0:
                                return
                            self._remaining -= 1
                        # Acquire before scheduling so a stalled host throttles
                        # itself rather than accumulating tasks.
                        await in_flight.acquire()
                        task = asyncio.create_task(issue(item))
                        pending.add(task)
                        task.add_done_callback(pending.discard)
                        # Space *issuance* by the interval. This is what keeps
                        # the request rate at the configured value; response
                        # latency no longer gates it.
                        await asyncio.sleep(interval)
            finally:
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

    async def _fetch_one(self, session: aiohttp.ClientSession, item: ClaimedUrl) -> None:
        """Fetch, extract and persist a single URL, updating frontier state.

        Args:
            session: Shared HTTP session.
            item: Claimed frontier entry.
        """
        if self.config.respect_robots_txt and not self.robots.allowed(item.host, item.url):
            self.frontier.mark(item.key, SKIPPED, http_status=None)
            self.stats.robots_skipped += 1
            return

        try:
            async with session.get(item.url) as resp:
                status = resp.status
                if status != 200:
                    self.stats.http_errors += 1
                    # 429/5xx are transient; 4xx are not worth retrying.
                    if status in (429, 500, 502, 503, 504):
                        self.frontier.requeue(item.key, self.config.max_retries)
                    else:
                        self.frontier.mark(item.key, FAILED, http_status=status)
                    return
                raw = await resp.read()
        except Exception:
            self.stats.exceptions += 1
            self.frontier.requeue(item.key, self.config.max_retries)
            return

        self.stats.fetched += 1
        self.stats.html_bytes += len(raw)
        html = raw.decode("utf-8", errors="ignore")

        # trafilatura is CPU-bound; keep it off the event loop.
        text, title = await asyncio.get_running_loop().run_in_executor(
            None, extract_text, html, item.url
        )

        if len(text) < self.config.min_chars:
            self.stats.too_short += 1
            self.frontier.mark(item.key, DONE, http_status=200, n_chars=len(text))
            return

        record = {
            "url": item.url,
            "host": item.host,
            "source": item.source,
            "source_type": self.source_type,
            "fetched_at": time.time(),
            "title": title,
            "text": text,
        }
        async with self._write_lock:
            self.writer.write(record)
            if self.raw_html_dir and self._rng.random() < self.config.raw_html_sample_rate:
                self._save_raw_sample(item, raw)

        self.stats.extracted += 1
        self.stats.chars_written += len(text)
        self.frontier.mark(item.key, DONE, http_status=200, n_chars=len(text))

    def _save_raw_sample(self, item: ClaimedUrl, raw: bytes) -> None:
        """Persist a gzipped copy of one page's raw HTML.

        The sample makes extraction reproducible and debuggable without keeping
        (or re-downloading) the full ~40 GB of HTML per language.

        Args:
            item: Frontier entry being sampled.
            raw: Raw response body.
        """
        name = item.key.hex() + ".html.gz"
        try:
            with gzip.open(Path(self.raw_html_dir) / name, "wb") as fh:
                fh.write(raw)
        except Exception as exc:
            LOGGER.debug("raw sample write failed for %s: %s", item.url, exc)


def host_of(url: str) -> str:
    """Return the hostname of a URL.

    Args:
        url: Absolute URL.

    Returns:
        Lowercased hostname, or ``""`` if the URL cannot be parsed.
    """
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def load_sources(config_path: Path | str) -> dict:
    """Load a language's ``sources.json`` inventory.

    Args:
        config_path: Path to the per-language source inventory.

    Returns:
        Parsed inventory dictionary.
    """
    with open(config_path, encoding="utf-8") as fh:
        return json.load(fh)


def config_from_sources(inventory: dict) -> CrawlConfig:
    """Build a :class:`CrawlConfig` from an inventory's ``crawl_defaults``.

    Args:
        inventory: Parsed ``sources.json`` contents.

    Returns:
        Configuration with inventory values applied over the defaults.
    """
    defaults = inventory.get("crawl_defaults", {})
    known = {f for f in CrawlConfig.__dataclass_fields__}
    return CrawlConfig(**{k: v for k, v in defaults.items() if k in known})
