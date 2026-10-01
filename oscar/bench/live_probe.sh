#!/bin/bash
# Live probe inside a running SkyRL job's allocation (srun --jobid=J --overlap): every INTERVAL s for
# DURATION s append to OUT: per-thread state + kernel wait channel of this user's ray/python
# processes, per-process GPU memory, GPU util, and PSI. Readable without ptrace.
# usage: live_probe.sh OUT [INTERVAL] [DURATION]
OUT=$1; INTERVAL=${2:-2}; DURATION=${3:-1800}
ulimit -u "$(ulimit -Hu)" 2>/dev/null
echo "probe start $(date) host=$(hostname) nproc soft=$(ulimit -u) hard=$(ulimit -Hu)" >> "$OUT"
end=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$end" ]; do
    {
        echo "=== $(date +%s.%N | cut -c1-14) $(date +%T)"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';'; echo
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';'; echo
        for p in memory cpu io; do [ -r /proc/pressure/$p ] && echo "psi-$p $(head -1 /proc/pressure/$p)"; done
        echo "user threads=$(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l) procs=$(ps -u "$USER" --no-headers 2>/dev/null | wc -l)"
        # threads not sleeping normally (R running, D uninterruptible) plus every thread's wchan of ray workers
        ps -u "$USER" -L -o pid,tid,stat,pcpu,wchan:28,comm --no-headers 2>/dev/null \
            | awk '$6 ~ /^(ray::|python|VLLM|pt_|cuda|EngineCore)/ || $3 ~ /^[RD]/' \
            | awk '$3 ~ /^[RD]/ || $4 > 5'
    } >> "$OUT"
    sleep "$INTERVAL"
done
