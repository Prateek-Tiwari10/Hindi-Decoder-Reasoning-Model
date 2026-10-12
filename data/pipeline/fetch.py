"""Crawl entry point: drain a language's frontier into compressed text shards.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

Run after :mod:`data.pipeline.discover` has populated the frontier.

Usage::

    python -m data.pipeline.fetch --sources data/pipeline/sources_hindi.json \\
                           --db artifacts/corpus/frontier.db \\
                           --out artifacts/corpus/manual \\
                           --max-pages 300000

This is the long pole of Phase 1 -- roughly nine hours per language at the
default politeness settings.  It is designed to be interrupted: frontier state
is committed as each URL completes, so re-running resumes from where it stopped
rather than restarting.  Output shards are never overwritten; a resumed run
opens a fresh shard index.

Progress is logged periodically so a long unattended run can be monitored with
``tail -f``, and the same numbers are available at any time from
``python -m data.pipeline.fetch --status --db <db>``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path

from data.pipeline.crawl import (
    CrawlConfig,
    PoliteCrawler,
    ShardWriter,
    config_from_sources,
    load_sources,
)
from data.pipeline.frontier import UrlFrontier

LOGGER = logging.getLogger("fetch")


async def _report_progress(crawler: PoliteCrawler, interval: float = 60.0) -> None:
    """Log crawl statistics periodically until cancelled.

    Args:
        crawler: Crawler whose statistics to report.
        interval: Seconds between reports.
    """
    while True:
        await asyncio.sleep(interval)
        LOGGER.info("progress | %s", crawler.stats.summary())


async def run(
    sources_path: Path,
    db_path: Path,
    out_dir: Path,
    max_pages: int | None,
    keep_raw: bool,
) -> None:
    """Crawl the frontier and write extracted documents.

    Args:
        sources_path: Language's ``sources.json``, used for politeness settings.
        db_path: Language's frontier database.
        out_dir: Directory for the output shards.
        max_pages: Optional ceiling on pages fetched this session.
        keep_raw: Retain the configured random sample of raw HTML for debugging.
    """
    inventory = load_sources(sources_path)
    config: CrawlConfig = config_from_sources(inventory)
    language = inventory["language"]

    frontier = UrlFrontier(db_path)
    counts = frontier.counts()
    LOGGER.info("language=%s frontier=%s", language, counts)
    if not counts.get("pending"):
        LOGGER.warning("no pending URLs - run data.pipeline.discover first")
        frontier.close()
        return

    writer = ShardWriter(out_dir, prefix="manual", max_bytes=config.shard_max_bytes)
    crawler = PoliteCrawler(
        frontier=frontier,
        writer=writer,
        config=config,
        source_type="manual",
        raw_html_dir=(Path(out_dir) / "raw_sample") if keep_raw else None,
    )

    LOGGER.info(
        "politeness: %.1f req/s/host, %d concurrent hosts, robots=%s",
        config.requests_per_second_per_host,
        config.max_concurrent_hosts,
        config.respect_robots_txt,
    )

    reporter = asyncio.create_task(_report_progress(crawler))

    # A Ctrl-C should stop cleanly: in-flight URLs return to PENDING on the next
    # run, and the current shard is closed properly rather than truncated.
    stopping = asyncio.Event()

    def _request_stop() -> None:
        """Signal handler: ask the crawl to wind down."""
        LOGGER.warning("stop requested - finishing in-flight requests")
        stopping.set()
        crawler._remaining = 0  # drains workers at their next loop check

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:  # not available on all platforms
            pass

    try:
        stats = await crawler.run(max_pages=max_pages)
    finally:
        reporter.cancel()
        writer.close()
        LOGGER.info("final | %s", crawler.stats.summary())
        LOGGER.info("frontier | %s", frontier.counts())
        LOGGER.info("per-host | host / done / pending / chars")
        for host, done, pending, chars in frontier.per_host_progress():
            LOGGER.info("  %-38s %8s %8s %12s", host, done, pending, f"{chars:,}")
        frontier.close()


def show_status(db_path: Path) -> None:
    """Print frontier progress without starting a crawl.

    Args:
        db_path: Language's frontier database.
    """
    with UrlFrontier(db_path, reset_in_flight=False) as frontier:
        print("state:", frontier.counts())
        print(f"{'host':<40} {'done':>9} {'pending':>9} {'chars':>14}")
        total_chars = 0
        for host, done, pending, chars in frontier.per_host_progress():
            total_chars += chars
            print(f"{host:<40} {done or 0:>9} {pending or 0:>9} {chars:>14,}")
        # ~10.7 bytes/token and ~2.68 bytes/char for Devanagari, so
        # tokens ~= chars / 4.0. Rough, but enough to track progress against
        # the 100M-token manual quota mid-crawl.
        print(f"\ntotal chars: {total_chars:,}  (~{total_chars / 4.0 / 1e6:.1f}M tokens est.)")


def main() -> None:
    """Parse arguments and run the crawl or status report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-pages", type=int, default=None,
                        help="stop after this many pages this session")
    parser.add_argument("--no-raw-sample", action="store_true",
                        help="do not retain the raw-HTML debug sample")
    parser.add_argument("--status", action="store_true",
                        help="print frontier progress and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.status:
        show_status(args.db)
        return
    if not args.sources or not args.out:
        raise SystemExit("--sources and --out are required unless --status is given")

    asyncio.run(run(args.sources, args.db, args.out, args.max_pages,
                    keep_raw=not args.no_raw_sample))


if __name__ == "__main__":
    main()
