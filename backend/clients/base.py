"""Shared HTTP plumbing for all provider clients: timeouts, retries with backoff,
concurrency caps and human-readable errors.

Every external call in the app goes through ``request_with_retry`` so a flaky
endpoint degrades into a clear message ("The voice service timed out") instead of
a stack trace in the UI.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

log = logging.getLogger("podcast.clients")

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ServiceError(RuntimeError):
    """An external service failed after retries. ``user_message`` is safe to show in the UI."""

    def __init__(self, service: str, user_message: str, detail: str = ""):
        super().__init__(f"{service}: {user_message} {detail}".strip())
        self.service = service
        self.user_message = user_message
        self.detail = detail


_client: httpx.AsyncClient | None = None


def http_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def request_with_retry(
    service: str,
    method: str,
    url: str,
    *,
    timeout: float,
    attempts: int = 3,
    semaphore: asyncio.Semaphore | None = None,
    **kwargs: Any,
) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if semaphore is not None:
                async with semaphore:
                    resp = await http_client().request(method, url, timeout=timeout, **kwargs)
            else:
                resp = await http_client().request(method, url, timeout=timeout, **kwargs)
        except httpx.TimeoutException as e:
            last = e
        except httpx.HTTPError as e:
            last = e
        else:
            if resp.status_code < 400:
                return resp
            err = ServiceError(service, f"returned HTTP {resp.status_code}", resp.text[:300])
            if resp.status_code not in RETRY_STATUS or attempt >= attempts:
                raise err
            last = err
        if attempt >= attempts:
            break
        wait = min(20.0, (2 ** (attempt - 1)) * 1.5) * (0.7 + 0.6 * random.random())
        log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs", service, attempt, attempts, last, wait)
        await asyncio.sleep(wait)
    if isinstance(last, httpx.TimeoutException):
        raise ServiceError(service, "timed out", str(last))
    if isinstance(last, ServiceError):
        raise last
    raise ServiceError(service, "is unreachable", str(last))
