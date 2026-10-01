# Shared configuration for running SkyRL on Oscar under Apptainer.
# Sourced by every 0*.sbatch script and by shell.sh. Not executable on its own.

# Code, scripts, logs, the SIF and any checkpoints live on /oscar/data (moved off purge-prone
# scratch 2026-10-01). Regenerable, file-count-heavy state (venv, uv cache, uv python, container
# HOME) lives on scratch under CACHE_ROOT -- /oscar/data has a group file-count quota; rebuild it
# with 24_rebuild_env_pinned_check.sbatch if a purge takes it.
SKYRL_ROOT=/oscar/data/deeptir/mborjigi/skyrl
CACHE_ROOT=/oscar/scratch/mborjigi/skyrl-cache

# --- container -------------------------------------------------------------
# The published SkyRL FSDP image. It is only a toolchain (Anyscale Ray 2.56.0 +
# CUDA 12.8 toolkit + uv + build-essential/libnuma) -- it contains no SkyRL
# source and no Python deps. Those are created on scratch by 02_setup_env.
SKYRL_IMAGE_URI="docker://novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8"
SIF="$SKYRL_ROOT/skyrl-ray2.56-py312-cu128.sif"

# --- source + venv ---------------------------------------------------------
SRC="$SKYRL_ROOT/SkyRL"
# main @ 2026-08-10. Pinned so the venv, the run scripts and the results all
# refer to one tree. Bump deliberately, then re-run 02_setup_env.
SKYRL_COMMIT=bce9ee9a80fbd262db44c79d5af12291ced5492d

# The venv lives outside the checkout: uv's docs recommend it, and it lets us
# delete/rebuild deps without disturbing the source tree.
VENV="$CACHE_ROOT/venv"

# Writable HOME inside the container. Deliberately NOT /home/ray -- binding
# over /home/ray would hide the image's own anaconda3 and ~/.local/bin/uv.
CHOME="$CACHE_ROOT/home"
UV_CACHE="$CACHE_ROOT/uv-cache"
UV_PY="$CACHE_ROOT/uv-python"

# uv is installed into the image at this fixed path (Dockerfile runs the
# astral installer as user `ray`). Referenced absolutely so we never depend on
# PATH surviving into the container.
UV=/home/ray/.local/bin/uv

# --- data (already staged on Oscar; nothing to download) -------------------
HF_CACHE=/oscar/data/deeptir/mborjigi/hf     # holds Qwen2.5-Coder-7B-Instruct (15G); moved off scratch 2026-10-01 (quota)
SQL_DATA=/oscar/data/deeptir/datasets/skyrl_sql
SQL_DB_PATH="$SQL_DATA/data"                 # OmniSQL db dirs: bird/ spider/ SynSQL-2.5M/ ...

# Checkpoints/exports live on the large /oscar/data partition, never scratch (scratch quota is
# 512G; old checkpoints there tripped the quota on 2026-10-01). Runs should save NONE unless the
# final model is needed for eval (trainer.ckpt_interval=-1, trainer.hf_save_interval=-1).
CKPT_ROOT=/oscar/data/deeptir/mborjigi/skyrl/ckpts
EXPORT_ROOT=/oscar/data/deeptir/mborjigi/skyrl/exports
LOG_ROOT="$SKYRL_ROOT/logs"

# ---------------------------------------------------------------------------
# skyrl_set_jobtmp -- pick node-local fast scratch for Ray's temp/spill dir.
#
# The GPU nodes report TmpDisk=0 and host /tmp is small and shared, so Ray's
# default /tmp/ray is a bad place for object spill. /jobtmp is Oscar's fast
# per-user job storage (1T quota). Falls back to scratch outside a job.
# ---------------------------------------------------------------------------
skyrl_set_jobtmp() {
    if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/jobtmp/$USER" ]; then
        JOBTMP="$(readlink -f /jobtmp/$USER)/skyrl-$SLURM_JOB_ID"
    else
        JOBTMP="$CACHE_ROOT/tmp/local-$$"
    fi
    mkdir -p "$JOBTMP/tmp" "$JOBTMP/ray"
    export JOBTMP
    # jobtmp carries a hard PER-USER file-count quota (2,000,000, shared across
    # every job on the account) as well as a block quota -- staging a SynSQL-2.5M
    # copy under $JOBTMP/db (285K+ files) on every run and never removing it
    # silently exhausts that quota after just a handful of jobs, at which point
    # EVERY subsequent job (this account's, on any node) fails at its own
    # "mkdir $JOBTMP" or "cp -r ... staging" step with "Disk quota exceeded" --
    # this is exactly what killed jobs 5133592/5134966 on 2026-08-21/22, root
    # cause was 6+ old runs' leftover $JOBTMP dirs, not a code bug. $JOBTMP is
    # unique per job (suffixed with $SLURM_JOB_ID), so it is always safe to
    # remove entirely once this job is done -- do that automatically here
    # rather than relying on every sbatch script to remember.
    trap 'rm -rf "$JOBTMP"' EXIT
}

# ---------------------------------------------------------------------------
# skyrl_prune_exports <export_path> [keep_n=2] -- delete all but the newest
# keep_n global_step_*/ HF export dirs under export_path.
#
# trainer.py's save_models() (the HF-format weights export used for eval) has
# no rotation of its own -- unlike save_checkpoints() (the resumable ckpt,
# which calls cleanup_old_checkpoints()/max_ckpts_to_keep after every save),
# confirmed by direct read of skyrl/train/trainer.py. Worse, is_epoch_end
# forces BOTH a ckpt save and an HF export at every epoch boundary regardless
# of ckpt_interval/hf_save_interval (same trigger bug the methodology doc
# calls out) -- with epochs=18 and 2 steps/epoch that's 18 unrotated HF
# exports over one run, ~15G each for the 7B model, ~270G/run with nothing
# to stop it. Call this periodically from a host-side loop around the
# training PID (export_path lives directly on /oscar/scratch, no container
# needed) and once more after training exits with keep_n=1 to leave only the
# true final checkpoint.
# ---------------------------------------------------------------------------
skyrl_prune_exports() {
    local export_path="$1" keep_n="${2:-2}" step n=0
    [ -d "$export_path" ] || return 0
    for step in $(find "$export_path" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
            | sed 's#.*/global_step_##' | sort -rn); do
        n=$((n + 1))
        if [ "$n" -gt "$keep_n" ]; then
            echo "  pruning stale export: $export_path/global_step_${step}"
            rm -rf "$export_path/global_step_${step}"
        fi
    done
}

# ---------------------------------------------------------------------------
# skyrl_run <command...> -- run a command inside the container.
#
# Set SKYRL_GPU=1 to pass --nv (GPU jobs only; --nv on a CPU node errors out).
# Requires JOBTMP (call skyrl_set_jobtmp first).
#
# --cleanenv is deliberate: the login shell's ~/.bashrc exports UV_CACHE_DIR,
# HF_HOME and friends pointing at the *host* layout, and leaking those into the
# container silently changes where the venv and model cache land. Everything
# the run needs is passed explicitly below.
# ---------------------------------------------------------------------------
skyrl_run() {
    local nvflag=()
    [ "${SKYRL_GPU:-0}" = "1" ] && nvflag=(--nv)

    mkdir -p "$CHOME" "$UV_CACHE" "$UV_PY" "$CKPT_ROOT" "$LOG_ROOT"

    # Only bind JOBTMP separately when it is outside the scratch tree we
    # already bind -- the non-Slurm fallback puts it under $SKYRL_ROOT, and
    # binding a path twice makes apptainer complain.
    local jtbind=()
    case "$JOBTMP" in
        /oscar/scratch/mborjigi/*) ;;
        *) jtbind=(--bind "$JOBTMP") ;;
    esac

    # HOME must be set with --home, not --env: apptainer explicitly refuses
    # `--env HOME=...` ("Overriding HOME ... is not permitted") and silently
    # leaves HOME pointing at the unmounted host home.
    #
    # Single-argument form, so $CHOME mounts at its own path and HOME is set to
    # it. Do NOT write `--home "$CHOME:/home/ray"` -- that would shadow the
    # image's own /home/ray, which holds anaconda3 and .local/bin/uv.
    #
    # UV_LINK_MODE=hardlink, not `copy` (the host bashrc's setting -- that
    # does not apply here): uv's `--isolated` builds materialise a private
    # venv per invocation under $UV_CACHE_DIR/builds-v0, and Ray spawns many
    # worker processes concurrently, each running its own `uv run
    # --isolated`. Under `copy` that is N simultaneous full multi-GB copies
    # over NFS, which silently ballooned $UV_CACHE_DIR until scratch went
    # over quota (job 4877123). $UV_CACHE_DIR and every build target share
    # one filesystem, so hardlinks are always valid and each materialisation
    # skips the data copy. NOTE: this fixes the disk bloat, not the
    # registration timeout below -- a single `uv run --isolated` invocation
    # measured 53-104s under hardlink mode too, so hardlink alone did not
    # make workers register fast enough (see RAY_worker_register_timeout_seconds).
    #
    # RAY_worker_register_timeout_seconds=300: every Ray worker process --
    # including a bare `@ray.remote` actor with no SkyRL-specific config,
    # confirmed by direct reproduction -- runs its own fresh `uv run
    # --isolated` (RAY_RUNTIME_ENV_HOOK below routes ALL worker startup
    # through uv, not just jobs that ask for it) to materialise 264 packages.
    # That consistently takes 53-104s even cache-warm, because on this NFS
    # filesystem uv's cost is dominated by per-file metadata round-trips, not
    # data volume -- hardlink mode above doesn't touch that. Ray's default
    # registration timeout (~60s) loses that race, especially with several
    # workers materialising concurrently and contending for the same NFS
    # metadata: raylet declares them dead ("process is dead, probably it
    # crashed during start" in raylet.err) and the actor never comes up (job
    # 4884148). This must be set before `ray start --head` runs (it is a
    # raylet-level config, not a client-side ray.init() setting), so it lives
    # here where it reaches every apptainer invocation, including the one
    # that starts the head.
    apptainer exec "${nvflag[@]}" --cleanenv \
        --home "$CHOME" \
        --bind /oscar/scratch/mborjigi \
        --bind /oscar/data/deeptir \
        "${jtbind[@]}" \
        --env TMPDIR="$JOBTMP/tmp" \
        --env UV_CACHE_DIR="$UV_CACHE" \
        --env UV_PROJECT_ENVIRONMENT="$VENV" \
        --env UV_PYTHON_INSTALL_DIR="$UV_PY" \
        --env UV_LINK_MODE=hardlink \
        --env RAY_worker_register_timeout_seconds=300 \
        --env HF_HOME="$HF_CACHE" \
        --env HF_HUB_ENABLE_HF_TRANSFER=1 \
        --env RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook \
        --env RAY_TMPDIR="$JOBTMP/ray" \
        --env RAY_USAGE_STATS_ENABLED=0 \
        --env NCCL_CUMEM_ENABLE=0 \
        --env NCCL_P2P_LEVEL=SYS \
        --env VLLM_USE_V1=1 \
        --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
        --env SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
        --env SKYRL_PROFILE_BWD="${SKYRL_PROFILE_BWD:-0}" \
        --env SKYRL_PROFILE_FWD="${SKYRL_PROFILE_FWD:-0}" \
        --env SKYRL_PROFILE_BWD_EVERY_N="${SKYRL_PROFILE_BWD_EVERY_N:-50}" \
        --env SKYRL_PROFILE_FWD_EVERY_N="${SKYRL_PROFILE_FWD_EVERY_N:-50}" \
        --env SKYRL_TRACE_RECORD="${SKYRL_TRACE_RECORD:-0}" \
        --env SKYRL_TRACE_PATH="${SKYRL_TRACE_PATH:-}" \
        --env SKYRL_STACK_SAMPLE_DIR="${SKYRL_STACK_SAMPLE_DIR:-}" \
        --env SKYRL_STACK_SAMPLE_INTERVAL="${SKYRL_STACK_SAMPLE_INTERVAL:-5}" \
        --env SKYRL_GC_DEBUG="${SKYRL_GC_DEBUG:-0}" \
        --env SKYRL_GC_DEBUG_MIN_S="${SKYRL_GC_DEBUG_MIN_S:-0.5}" \
        --env SKYRL_GC_FREEZE="${SKYRL_GC_FREEZE:-0}" \
        "$SIF" "$@"
}

# ---------------------------------------------------------------------------
# skyrl_check_egress -- fail fast if the node cannot reach the package indexes.
# `uv sync` needs PyPI, download.pytorch.org and flashinfer.ai; a compute node
# without egress produces a confusing resolver error many minutes in.
# ---------------------------------------------------------------------------
skyrl_check_egress() {
    local ok=0 h
    # github.com matters too: uv fetches the pinned CPython from
    # python-build-standalone releases there, and we clone SkyRL from it.
    for h in https://pypi.org/simple/ https://download.pytorch.org/whl/cu128 \
             https://huggingface.co https://github.com https://flashinfer.ai/whl/cu128 ; do
        if curl -sSf --max-time 20 -o /dev/null "$h"; then
            echo "  egress OK: $h"
        else
            echo "  EGRESS FAIL: $h"; ok=1
        fi
    done
    if [ "$ok" != 0 ]; then
        echo "ERROR: this node has no outbound access to the package indexes." >&2
        echo "       Re-run the setup step somewhere with egress, or pre-warm" >&2
        echo "       \$UV_CACHE_DIR and \$HF_HOME from a node that has it." >&2
        return 1
    fi
}

