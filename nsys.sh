#!/bin/zsh

SKYRL_DIR=$(dirname $(realpath $0))
ROOT_OUT_DIR=/local_nvme1/mborjigi/output

source $SKYRL_DIR/.venv/bin/activate

which python

# EVIL: arguments for gpu-metrics-devices are integer IDs but DIFFERENT from those returned by nvidia-smi. what
# we think of as gpus 0,1,2,3 are in fact 1,0,3,2 to nsys. Find out by calling sudo nsys profile --gpu-metrics-devices=help and comparing it to nvidia-smi. terrible.
# export CUDA_VISIBLE_DEVICES=1,2,3
# export CUDA_LAUNCH_BLOCKING=1
export TMPDIR=/local_nvme1/mborjigi/tmp
export DEEPSPEED_TIMEOUT=60
export RAY_NCCL_TIMEOUT=120
export NCCL_TIMEOUT=120
export NCCL_DEBUG=TRACE
export TORCH_DISTRIBUTED_DEBUG=INFO
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_BACKEND_LOG_LEVEL=info
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=true
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=72506c98349d6bcd32b4e33eec7b5513453c1502
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.13.0
# Disables the dashboard and its associated metrics agent
export RAY_INCLUDE_DASHBOARD=0
# Prevents Ray from trying to export metrics to the agent
export RAY_metrics_export_port=0
# Optional: Silences Ray's internal logging for cleaner output
export RAY_SCHEDULER_EVENTS=0

OUT_DIR=$ROOT_OUT_DIR/$1
SCRIPT=$SKYRL_DIR/examples/train/text_to_sql/run_skyrl_sql_fast_debug_1gpu.sh
SCRIPT=$SKYRL_DIR/examples/train/text_to_sql/run_skyrl_sql_fast_debug_3b_4gpu.sh
SCRIPT=$SKYRL_DIR/examples/train/text_to_sql/run_skyrl_sql_fast_debug_7b_4gpu.sh
SCRIPT=$SKYRL_DIR/examples/train/mini_swe_agent/run_mini_swe_8B.sh
SCRIPT=$SKYRL_DIR/examples/train/mini_swe_agent/run_mini_swe_30B.sh
SCRIPT=$SKYRL_DIR/examples/train/text_to_sql/run_skyrl_sql.sh
# SCRIPT=$SKYRL_DIR/examples/train/gsm8k/run_gsm8k.sh

rm -rf $OUT_DIR
mkdir $OUT_DIR
mkdir -p $ROOT_OUT_DIR/aux_logs
echo "dumping output here: $OUT_DIR/output.log"
echo "running..."
cat $SCRIPT > $OUT_DIR/script.sh
cat $0 > $OUT_DIR/nsys_script.sh

# Start CPU usage monitoring, thanks to gemini
# (
#   echo "Timestamp,PID,CPU%,MEM%,Command" > $OUT_DIR/cpu_metrics.csv
  
#   while true; do
#     timestamp=$(date +"%Y-%m-%d %H:%M:%S.%3N")
#     ps -u "$USER" -o "pid=,pcpu=,pmem=,args=" -ww --no-headers 2>/dev/null | awk -v ts="$timestamp" '{print ts "," $1 "," $2 "," $3 "," $4}' >> "$OUT_DIR/cpu_metrics.csv"
#     sleep 0.1
#   done
# ) &
# cpu_monitor_pid=$!
        
nvidia-smi dmon -s umt -o DT -f $OUT_DIR/gpu_metrics.csv &
gpu_monitor_pid=$!
trap "echo 'Killing monitors...'; kill $cpu_monitor_pid $gpu_monitor_pid 2>/dev/null;" INT TERM HUP QUIT PIPE EXIT

nsys profile --gpu-metrics-devices=1,0,3,2 --cuda-memory-usage=false --trace=cuda --cpuctxsw=process-tree --gpu-metrics-frequency=100 --force-overwrite=false --sample=process-tree --samples-per-backtrace=32 --wait=primary --delay=90 -o $OUT_DIR/nsys_report -- zsh $SCRIPT > $OUT_DIR/output.log 2>&1
kill $cpu_monitor_pid $gpu_monitor_pid 2>/dev/null

echo "Done! Exporting to sqlite..."
nsys export --type=sqlite --ts-normalize=true --force-overwrite=false --output=$OUT_DIR/nsys_report.sqlite $OUT_DIR/nsys_report.nsys-rep > /dev/null 2>&1

echo "Indexing sqlite..."
sqlite3 $OUT_DIR/nsys_report.sqlite \
  "CREATE INDEX IF NOT EXISTS idx_gpu_metrics_metricId_typeId ON gpu_metrics(metricId, typeId);"

mv $ROOT_OUT_DIR/aux_logs "$OUT_DIR/"

mv $ROOT_OUT_DIR/*.json* "$OUT_DIR/"
echo "Done! merging profiler jsons..."

# Note: have to activate the profiler environment for merge_profiles.py
deactivate
source .venv/bin/activate
python3 merge_profiles.py $1
echo "Output dir: $OUT_DIR/output.log"
echo "Output log: $OUT_DIR/output.log"
