import sys
import glob
import os
import re
import collections
import orjson as json
from joblib import Parallel, delayed
output_dir = sys.argv[1]  # e.g., /local_nvme1/mborjigi/output
output_dir = os.path.join("/local_nvme1/mborjigi/output", sys.argv[1])
trace_files = glob.glob(f"{output_dir}/*.json*")
trace_to_files = collections.defaultdict(list)
for trace_file in trace_files:
    base_name = os.path.basename(trace_file)
    match = re.match(r"^(.*)\.json(?:\.(\d+))?$", base_name)
    if match:
        profile_name = match.group(1)
        trace_to_files[profile_name].append(trace_file)

def merge_trace_files(trace, files, output_dir):
    print(f"Merging trace files for profile: {trace}...")
    print(f"Merged files: {files}")
    combined_trace = {"traceEvents": []}
    for file in files:
        with open(file, 'r') as f:
            try:
                data = json.loads(f.read())
            except json.JSONDecodeError as e:
                print(f"    Error decoding JSON from file {file}: {e}")
                continue
            if "traceEvents" in data:
                combined_trace["traceEvents"].extend(data["traceEvents"])
            else:
                combined_trace["traceEvents"].extend(data)

    with open(os.path.join(output_dir, f"{trace}.json"), 'w') as out_f:
        out_f.write(json.dumps(combined_trace).decode('utf-8'))
    print(f"Saved merged trace to {os.path.join(output_dir, f'{trace}.json')}")

Parallel(n_jobs=-1)(delayed(merge_trace_files)(trace, files, output_dir) for trace, files in trace_to_files.items())
