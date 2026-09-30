"""Opt-in periodic stack sampler for diagnosing silent stalls.

When ``SKYRL_STACK_SAMPLE_DIR`` is set, :func:`start_stack_sampler` starts a daemon thread that
every ``SKYRL_STACK_SAMPLE_INTERVAL`` seconds (default 5) appends the calling process's main-thread
Python stack (innermost frames) to ``<dir>/<tag>-<pid>.log``, one timestamped line per sample.
Useful where py-spy cannot attach (kernel.yama.ptrace_scope >= 2). No-op when the variable is
unset; safe to call more than once per process.
"""

import os
import sys
import threading
import time
import traceback

_started = False


def start_stack_sampler(tag: str) -> None:
    global _started
    out_dir = os.environ.get("SKYRL_STACK_SAMPLE_DIR")
    if not out_dir or _started:
        return
    _started = True
    interval = float(os.environ.get("SKYRL_STACK_SAMPLE_INTERVAL", "5"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}-{os.getpid()}.log")
    main_ident = threading.main_thread().ident

    me = threading.get_ident()

    def run():
        with open(path, "a", buffering=1) as f:
            while True:
                time.sleep(interval)
                names = {t.ident: t.name for t in threading.enumerate()}
                now = time.time()
                for ident, frame in sys._current_frames().items():
                    if ident == me:
                        continue
                    stack = traceback.extract_stack(frame)[-8:]
                    where = " <- ".join(
                        f"{os.path.basename(s.filename)}:{s.name}:{s.lineno}" for s in reversed(stack)
                    )
                    tname = "main" if ident == main_ident else names.get(ident, str(ident))
                    f.write(f"{now:.1f} [{tname}] {where}\n")

    threading.Thread(target=run, name="skyrl-stack-sampler", daemon=True).start()
