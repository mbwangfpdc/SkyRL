import os
traj_dir = "/local_nvme1/mborjigi/tmp/mini_swe_agent_trajs"

for root, dirs, files in os.walk(traj_dir):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file), 'r') as f:
                import json
                traj = json.load(f)
                # print("exit status: ", traj["info"]["exit_status"])
                print("exit reward: ", traj["reward"])
                # print("eval error: ", traj["eval_error"])
                # for step in traj["steps"]:
                #     if "reward" not in step:
                #         print(f"Missing reward in {file}")
