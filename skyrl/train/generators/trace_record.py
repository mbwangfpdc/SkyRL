"""Deterministic-rollout trace recording, compatible with granular-cais-rl's replay format.

granular-cais-rl (a separate reimplementation of this same SkyRL-SQL task, used for a
systems comparison) can *replay* a recorded rollout trace to pin the generation workload
(decode lengths, per-turn observation sizes, env service times) and isolate training-side
(framework) latency differences from run-to-run generation variance. This module makes SkyRL
capable of *emitting* a trace in that exact JSONL schema, so a SkyRL-recorded trace can be fed
into granular's replay consumer (``[replay] mode = "replay"``) directly.

Schema (mirrors granular-cais-rl's ``src/granular_cais_rl/replay.py``):

  Header line:  {"kind": "header", "fingerprint": {...}, ...meta}
  One line per training trajectory:
    {"step", "mb", "session_id", "sample_id", "prompt_tokens", "env_init_service_s",
     "turns": [{"gen_tokens", "obs_tokens", "env_step_service_s", "reward", "done",
                "finish_reason", "obs_messages"}, ...],
     "reward", "stop_reason", "response_tokens"}

Only eval trajectories are excluded (same convention as granular: eval is a different
workload and would collide with the training (step, session_id) keyspace).

``session_id`` IS SkyRL's own ``f"{instance_id}_{repetition_id}"`` (``instance_id`` is a
dataset-wide UID -- see ``TrajectoryID``). This is deliberate: keying by a stable per-row
identity rather than a batch position means a granular-side replay run reads its own batch
composition directly out of the trace (see granular's ``TrainManager._next_batch_from_trace``)
instead of independently iterating its own dataset and hoping the two frameworks' shuffles
happen to agree -- they don't have to agree at all, on either side, for replay to address the
same underlying row. (An earlier version of this trace format used ``TrajectoryID.sample_index``
-- a batch position -- for exactly this purpose, which required disabling shuffling on this
side; that's no longer necessary.)

``obs_messages`` on each turn is the actual recorded observation (the real chat messages the
env returned, not a token count) so a replay run can hand the model the same real content
rather than synthesized filler -- filler observations can themselves derail generation in ways
uncorrelated with either framework's real behavior, and can't reproduce whatever prefix-cache
hits the real content would have gotten.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Config fields that define the WORKLOAD, keyed to match granular-cais-rl's
# FINGERPRINT_FIELDS names exactly (see replay.py) so its TraceIndex.validate() can check
# them. Fields with no clean cross-system value match (e.g. train_data: same content, different
# path on disk) are omitted here and recorded in `meta` instead, which granular's validate()
# never checks.
FINGERPRINT_FIELDS = (
    "model_name",
    "seed",
    "batch_size",
    "mini_batch_size",
    "num_generations",
    "max_turns",
    "max_prompt_len",
    "max_generate_length",
    "max_input_length",
    "use_conversation_multi_turn",
    "end_tags",
)


def make_fingerprint(cfg) -> Dict[str, Any]:
    """Build a granular-cais-rl-compatible fingerprint dict from SkyRL's top-level config."""
    return {
        "model_name": cfg.trainer.policy.model.path,
        "seed": cfg.trainer.seed,
        "batch_size": cfg.trainer.train_batch_size,
        "mini_batch_size": cfg.trainer.policy_mini_batch_size,
        "num_generations": cfg.generator.n_samples_per_prompt,
        "max_turns": cfg.generator.max_turns,
        "max_prompt_len": cfg.trainer.max_prompt_length,
        "max_generate_length": cfg.generator.sampling_params.max_generate_length,
        "max_input_length": cfg.generator.max_input_length,
        "use_conversation_multi_turn": cfg.generator.use_conversation_multi_turn,
        "end_tags": list(cfg.generator.sampling_params.stop or []),
    }


@dataclass
class TurnTrace:
    gen_tokens: int
    obs_tokens: int
    env_step_service_s: float
    reward: float
    done: bool
    finish_reason: str = ""
    # Real observation chat messages ([{"role": ..., "content": ...}]), replayed
    # verbatim on the granular side rather than synthesized from obs_tokens alone.
    obs_messages: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TrajectoryTrace:
    step: int
    mb: int
    session_id: str
    sample_id: int
    prompt_tokens: int
    env_init_service_s: float
    turns: List[TurnTrace] = field(default_factory=list)
    reward: float = 0.0
    stop_reason: str = ""
    response_tokens: int = 0

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["turns"] = [t.__dict__ for t in self.turns]
        return d


class TraceRecorder:
    """Append-only JSONL writer for rollout traces, in granular-cais-rl's exact schema.

    SkyRL's agent_loop tasks all run concurrently on the driver event loop (asyncio, single
    process/thread for the Python-level work -- only the inference-engine call and env.step
    cross into other threads/processes), so a lock around the file write is enough; no
    cross-process coordination is needed.
    """

    def __init__(self, path: str, fingerprint: Dict[str, Any], meta: Optional[Dict[str, Any]] = None):
        self.path = path
        self._lock = threading.Lock()
        self._n = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", buffering=1)
        if fresh:
            header = {"kind": "header", "fingerprint": fingerprint}
            header.update(meta or {})
            self._fh.write(json.dumps(header) + "\n")

    def write(self, traj: TrajectoryTrace) -> None:
        line = json.dumps(traj.to_json())
        with self._lock:
            self._fh.write(line + "\n")
            self._n += 1

    @property
    def count(self) -> int:
        return self._n

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def build_trace_recorder_from_env(cfg) -> Optional[TraceRecorder]:
    """Construct a TraceRecorder from SKYRL_TRACE_RECORD / SKYRL_TRACE_PATH, or None if unset.

    Kept as an env-var toggle (matching this fork's existing SKYRL_PROFILE_* instrumentation
    convention) rather than a new config schema field, since this is experimental
    instrumentation, not a supported training feature.
    """
    if os.environ.get("SKYRL_TRACE_RECORD", "0") != "1":
        return None
    path = os.environ.get("SKYRL_TRACE_PATH") or os.path.join(cfg.trainer.log_path, "rollout_trace.jsonl")
    fingerprint = make_fingerprint(cfg)
    meta = {
        "train_data": list(cfg.data.train_data),
        "run_name": cfg.trainer.run_name,
        "source": "skyrl",
    }
    return TraceRecorder(path, fingerprint, meta)
