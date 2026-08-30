"""Cycle 4 -- minimal structured request logging for the inference API.

Logs method/path/status/latency and prediction outcome (predicted class +
is_malicious). Never logs raw feature values -- the flow feature vectors
are the sensitive input here.
"""
from __future__ import annotations

import logging
import time

from fastapi import Request

logger = logging.getLogger("ids_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}')
    )
    logger.addHandler(handler)


async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        '{"method": "%s", "path": "%s", "status": %d, "latency_ms": %.2f}',
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response
