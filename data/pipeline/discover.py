"""URL discovery: expand a language's source inventory into a crawl frontier.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

Run this before :mod:`data.pipeline.fetch`.  It reads ``<lang>/configs/sources.json``,
expands every site's sitemaps into article URLs, and loads them into that
language's frontier database.

Discovery is deliberately a separate stage from fetching.  It is fast (minutes),
network-light, and its output is exactly the thing worth inspecting before
committing to nine hours of crawling: if a site yields far fewer URLs than
expected, that shows up here rather than halfway through the crawl.

Usage::

    python -m data.pipeline.discover --sources data/pipeline/sources_hindi.json \\
                              --db artifacts/corpus/frontier.db

Re-running is safe and cheap: the frontier deduplicates by URL, so a second run
adds only URLs that are genuinely new.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import aiohttp

from data.pipeline.crawl import SitemapEnumerator, config_from_sources, host_of, load_sources
from data.pipeline.frontier import UrlFrontier

LOGGER = logging.getLogger("discover")


class SiteQuotaReached(Exception):
    """Raised to stop expanding a site once its URL quota is filled."""


async def discover_site(
    session: aiohttp.ClientSession,
    site: dict,
    frontier: UrlFrontier,
    max_urls: int | None = None,
) -> tuple[str, int, int]:
    """Expand one site's sitemaps and load the URLs into the frontier.

    Args:
        session: Shared HTTP session.
        site: One entry from the inventory's ``sites`` list.
        frontier: Destination frontier.
        max_urls: Stop after adding this many URLs from this site.

            This cap exists for corpus quality, not just runtime.  Sitemap
            archives differ in size by orders of magnitude -- Amar Ujala's index
            expands to 6.8M URLs while other Hindi outlets yield tens of
            thousands -- so uncapped discovery produces a corpus that is
            effectively one newspaper.  Capping each site keeps register, topic
            and editorial diversity spread across all sources, and makes
            deduplication more effective because near-duplicate syndicated
            stories are less concentrated.

    Returns:
        Tuple of ``(site_name, urls_found, urls_newly_added)``.
    """
    name = site.get("name", site["host"])
    sitemaps = site.get("sitemaps") or []
    if not sitemaps:
        LOGGER.info("%-24s no sitemaps configured (discovery=%s) - skipping",
                    name, site.get("discovery"))
        return name, 0, 0

    added = 0

    def commit(urls: list[str]) -> None:
        """Persist one child sitemap's URLs immediately.

        Committing per child sitemap rather than per site matters for large
        archives: BBC's index alone fans out into 122 child sitemaps taking
        roughly an hour to walk. Buffering until the site finished would risk
        losing all of it to an interruption, and would block the crawl from
        starting on URLs that are already known.

        Args:
            urls: Page URLs from one child sitemap.
        """
        nonlocal added
        # A sitemap may legitimately point at a different subdomain (CDN,
        # regional edition). Bucket by actual host so rate limiting stays
        # correct per server.
        by_host: dict[str, list[str]] = {}
        for url in urls:
            by_host.setdefault(host_of(url) or site["host"], []).append(url)
        for host, host_urls in by_host.items():
            added += frontier.add_many(host_urls, host=host, source=name)
        if max_urls is not None and added >= max_urls:
            raise SiteQuotaReached

    enumerator = SitemapEnumerator(session)
    try:
        urls = await enumerator.enumerate(
            sitemaps, url_filter=site.get("url_filter"), on_urls=commit
        )
        found = len(urls)
    except SiteQuotaReached:
        # The quota is a stopping condition, not an error: everything discovered
        # so far is already committed by `commit`.
        found = added
        LOGGER.info("%-24s quota of %d reached - stopping expansion", name, max_urls)

    LOGGER.info("%-24s found=%-8d new=%-8d", name, found, added)
    return name, found, added


async def run(
    sources_path: Path,
    db_path: Path,
    only: str | None = None,
    max_urls_per_site: int | None = None,
) -> None:
    """Discover URLs for every site in an inventory.

    Args:
        sources_path: Path to the language's ``sources.json``.
        db_path: Path to the language's frontier database.
        only: Optional hostname; restricts discovery to that single site.
        max_urls_per_site: Per-site URL cap, for source balance.
    """
    inventory = load_sources(sources_path)
    config = config_from_sources(inventory)
    sites = inventory["sites"]
    if only:
        sites = [s for s in sites if s["host"] == only]
        if not sites:
            raise SystemExit(f"no site with host={only!r} in {sources_path}")

    LOGGER.info("language=%s sites=%d", inventory["language"], len(sites))

    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=8, ssl=False)
    headers = {"User-Agent": config.user_agent}

    with UrlFrontier(db_path) as frontier:
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector, headers=headers
        ) as session:
            # Sites are expanded sequentially: sitemap indexes fan out into
            # hundreds of child requests, and running several sites at once would
            # breach the per-host politeness budget during discovery itself.
            results = []
            for site in sites:
                try:
                    results.append(
                        await discover_site(session, site, frontier, max_urls_per_site)
                    )
                except Exception as exc:
                    LOGGER.error("%s failed: %s", site.get("name", site["host"]), exc)

        total_found = sum(r[1] for r in results)
        total_new = sum(r[2] for r in results)
        LOGGER.info("-" * 62)
        LOGGER.info("discovered=%d new=%d", total_found, total_new)
        LOGGER.info("frontier state: %s", frontier.counts())


def main() -> None:
    """Parse arguments and run discovery."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sources", required=True, type=Path,
                        help="path to <lang>/configs/sources.json")
    parser.add_argument("--db", required=True, type=Path,
                        help="path to the language's frontier database")
    parser.add_argument("--only", default=None,
                        help="restrict discovery to a single host")
    parser.add_argument("--max-urls-per-site", type=int, default=None,
                        help="cap URLs taken from each site, for source balance")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(run(args.sources, args.db, args.only, args.max_urls_per_site))


if __name__ == "__main__":
    main()
