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
import re
from typing import Any

import httpx

log = logging.getLogger("podcast.clients")

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_WAIT_S = 90.0  # longer server-requested waits are reported instead of slept through
STATUS_MESSAGES = {
    400: "rejected the request (HTTP 400)",
    401: "rejected the API key (HTTP 401) - check the key in the secrets file",
    403: "refused access (HTTP 403) - check the API key and its permissions",
    404: "endpoint or model was not found (HTTP 404)",
    429: "is rate-limiting us (HTTP 429) - try again in a minute",
}


class ServiceError(RuntimeError):
    """An external service failed after retries. ``user_message`` is safe to show in the UI."""

    def __init__(self, service: str, user_message: str, detail: str = "", status: int | None = None):
        super().__init__(f"{service}: {user_message} {detail}".strip())
        self.service = service
        self.user_message = user_message
        self.detail = detail
        self.status = status  # HTTP status if the service answered with an error, else None
        self.retry_after: float | None = None  # seconds the service asked us to wait, if any


def retry_delay(resp: httpx.Response) -> float | None:
    """Server-requested wait: a Retry-After header, or Google's RetryInfo ("retryDelay": "13s")."""
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', resp.text[:4000])
    return float(m.group(1)) if m else None


_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def http_client() -> httpx.AsyncClient:
    """Shared connection pool, recreated if the event loop changed (an AsyncClient is bound to its loop)."""
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = httpx.AsyncClient(limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))
        _client_loop = loop
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and _client_loop is asyncio.get_running_loop():
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
    server_wait: float | None = None
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
            err = ServiceError(service, STATUS_MESSAGES.get(resp.status_code, f"returned HTTP {resp.status_code}"), resp.text[:300],
                               status=resp.status_code)
            if resp.status_code not in RETRY_STATUS or attempt >= attempts:
                raise err
            last = err
            server_wait = retry_delay(resp)
            if server_wait is not None and server_wait > MAX_WAIT_S:
                # e.g. a daily quota that resets in hours: waiting is pointless, let the caller
                # switch to another model (or tell the user) right away.
                err.retry_after = server_wait
                raise err
        if attempt >= attempts:
            break
        wait = min(20.0, (2 ** (attempt - 1)) * 1.5) * (0.7 + 0.6 * random.random())
        if server_wait is not None:
            wait = min(60.0, server_wait + random.random())
            server_wait = None
        log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs", service, attempt, attempts, last, wait)
        await asyncio.sleep(wait)
    if isinstance(last, httpx.TimeoutException):
        raise ServiceError(service, "timed out", str(last))
    if isinstance(last, ServiceError):
        raise last
    raise ServiceError(service, "is unreachable", str(last))
