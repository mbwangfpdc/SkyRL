#!/bin/bash
# Host-side sampler for SkyRL stall debugging: every INTERVAL s append to OUT/host.log the job
# cgroup's memory usage/limit and memory.stat (rss/cache/shmem), system PSI (memory/cpu/io), GPU
# util/memory, and this user's top CPU processes.  Usage: host_sampler.sh OUT [INTERVAL]
OUT=$1; INTERVAL=${2:-5}
mkdir -p "$OUT"
memcg=$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3}' /proc/self/cgroup)
MC=/sys/fs/cgroup/memory$memcg
while true; do
    {
        echo "=== $(date +%s.%N | cut -c1-14) $(date +%T)"
        if [ -r "$MC/memory.usage_in_bytes" ]; then
            echo "cgroup usage_gb=$(( $(cat "$MC/memory.usage_in_bytes") / 1073741824 )) limit_gb=$(( $(cat "$MC/memory.limit_in_bytes") / 1073741824 ))" \
                "$(grep -E '^(total_rss|total_cache|total_shmem|total_mapped_file|total_pgmajfault) ' "$MC/memory.stat" | tr '\n' ' ')"
        fi
        for p in memory cpu io; do [ -r /proc/pressure/$p ] && echo "psi-$p $(tr '\n' ' ' < /proc/pressure/$p)"; done
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';'; echo
        ps -u "$USER" -o pid,pcpu,rss,stat,comm --sort=-pcpu | head -8
    } >> "$OUT/host.log"
    sleep "$INTERVAL"
done
