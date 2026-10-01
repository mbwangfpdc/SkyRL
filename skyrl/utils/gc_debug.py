"""Opt-in GC diagnostics for the SkyRL pre-training stall.

``SKYRL_GC_DEBUG=1``: log every Python garbage collection that takes >= ``SKYRL_GC_DEBUG_MIN_S``
seconds (default 0.5) with its generation and duration, and log a one-time map of Python thread
names to OS thread ids (so a busy thread seen in ``ps -L`` can be named).
``SKYRL_GC_FREEZE=1``: :func:`freeze_after_init` calls ``gc.freeze()``, moving every object that
exists at that point (model, modules, imports) to the permanent generation so later full
collections only scan objects created afterwards.
"""

import gc
import os
import threading
import time

from loguru import logger

_installed = False


def install(tag: str) -> None:
    global _installed
    if _installed or os.environ.get("SKYRL_GC_DEBUG") != "1":
        return
    _installed = True
    min_s = float(os.environ.get("SKYRL_GC_DEBUG_MIN_S", "0.5"))
    state = {}

    def cb(phase, info):
        if phase == "start":
            state["t"] = time.time()
            return
        t0 = state.pop("t", None)
        if t0 is None:
            return
        dt = time.time() - t0
        if dt >= min_s:
            logger.info(
                f"[gc-debug] {tag} gen{info.get('generation')} collection took {dt:.2f}s "
                f"(collected {info.get('collected')}, thread {threading.current_thread().name} "
                f"tid {threading.get_native_id()})"
            )

    gc.callbacks.append(cb)
    logger.info(f"[gc-debug] {tag} installed; gc thresholds {gc.get_threshold()}")


def log_threads(tag: str) -> None:
    if os.environ.get("SKYRL_GC_DEBUG") != "1":
        return
    names = ", ".join(f"{t.name}={t.native_id}" for t in threading.enumerate())
    logger.info(f"[gc-debug] {tag} pid {os.getpid()} threads: {names}")


def freeze_after_init(tag: str) -> None:
    if os.environ.get("SKYRL_GC_FREEZE") != "1":
        return
    gc.collect()
    gc.freeze()
    logger.info(f"[gc-debug] {tag} gc.freeze(): {gc.get_freeze_count()} objects frozen")
