# ============================================================
# DOC AI Vision Service – Crawler Scheduler
# ============================================================

import asyncio
import logging
from datetime import datetime
from typing import Optional

import croniter  # pip install croniter

from src.common.config import get_config, VisionConfig
from src.crawler.fetcher import CrawlerFetcher
from src.crawler.parser import CrawlerParser

logger = logging.getLogger(__name__)


class CrawlerScheduler:
    """
    Cron‑based scheduler for the automated web crawler.

    Reads the crawler configuration (enabled, schedule, max_items_per_run)
    and runs a continuous loop that triggers a crawl at each scheduled time.
    The scheduler runs as an asyncio task in the background.
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        """
        Initialize the scheduler.

        Args:
            config: VisionConfig instance. If None, loads the global config.
        """
        self.config = config or get_config()
        self.crawler_config = self.config.crawler
        self.enabled = self.crawler_config.enabled
        self.schedule = self.crawler_config.schedule
        self.max_items = self.crawler_config.max_items_per_run
        self.sources = self.crawler_config.sources

        self.fetcher = CrawlerFetcher(self.config)
        self.parser = CrawlerParser(self.config)

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.logger = logging.getLogger(f"{__name__}.CrawlerScheduler")

        if not self.enabled:
            self.logger.info("Crawler is disabled in configuration.")

    async def _run_once(self) -> None:
        """
        Perform a single crawl iteration: fetch and parse data from all sources.
        """
        self.logger.info("Starting scheduled crawl")
        total_items = 0
        for source in self.sources:
            try:
                self.logger.info(f"Fetching from source: {source.name} ({source.url})")
                # Fetch raw data (e.g., JSON, image list, etc.)
                data = await self.fetcher.fetch(source)
                if not data:
                    self.logger.warning(f"No data returned from {source.name}")
                    continue

                # Parse and validate the data
                items = await self.parser.parse(source, data)
                self.logger.info(f"Parsed {len(items)} items from {source.name}")

                # Store/process items (e.g., download images, store metadata)
                # The parser should handle storage via MinIO/PostgreSQL.
                # We'll leave that to the parser's implementation.
                total_items += len(items)

                # Stop if we've reached the max per run
                if self.max_items and total_items >= self.max_items:
                    self.logger.info(f"Reached max items per run ({self.max_items}), stopping early")
                    break
            except Exception as e:
                self.logger.error(f"Error processing source {source.name}: {e}", exc_info=True)

        self.logger.info(f"Completed crawl: processed {total_items} items total")

    async def _loop(self) -> None:
        """
        Main scheduler loop: sleep until the next cron trigger, then run.
        """
        if not self.enabled:
            self.logger.info("Scheduler loop not started (crawler disabled)")
            return

        cron = croniter.croniter(self.schedule, datetime.now())
        self.logger.info(f"Scheduler started with cron schedule: {self.schedule}")

        while self._running:
            now = datetime.now()
            next_time = cron.get_next(datetime)
            wait_seconds = (next_time - now).total_seconds()
            if wait_seconds > 0:
                self.logger.debug(f"Sleeping {wait_seconds:.1f} seconds until next crawl")
                await asyncio.sleep(wait_seconds)
            # Run the crawl
            await self._run_once()

    def start(self) -> None:
        """
        Start the scheduler as a background asyncio task.
        """
        if not self.enabled:
            self.logger.info("Scheduler start requested but crawler is disabled.")
            return
        if self._task is not None and not self._task.done():
            self.logger.warning("Scheduler already running.")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop())
        self.logger.info("Crawler scheduler started in background")

    async def stop(self) -> None:
        """
        Stop the scheduler gracefully.
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.logger.info("Crawler scheduler stopped")

    def is_running(self) -> bool:
        """Return True if the scheduler task is currently running."""
        return self._task is not None and not self._task.done()