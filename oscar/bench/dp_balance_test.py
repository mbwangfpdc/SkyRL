"""CPU check of RayPPOTrainer._balance_dp_tokens (trainer.dp_token_balance) on a synthetic batch."""
import random
from types import SimpleNamespace

import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.trainer import RayPPOTrainer

random.seed(0)
N, S, DP = 1280, 64, 4
lens = [random.choice([8] * 3 + list(range(10, S))) for _ in range(N)]
am = torch.zeros(N, S, dtype=torch.long)
for i, L in enumerate(lens):
    am[i, S - L:] = 1
seq = torch.arange(N).unsqueeze(1).repeat(1, S)  # row i carries its original index everywhere
data = TrainingInputBatch({"sequences": seq, "attention_mask": am, "advantages": seq.float()})
uids = [f"p{i // 5}" for i in range(N)]
data.metadata = {"uids": list(uids), "policy_mini_batch_boundaries": [(0, 640), (640, 1280)]}

fake = SimpleNamespace(
    cfg=SimpleNamespace(trainer=SimpleNamespace(
        algorithm=SimpleNamespace(loss_reduction="token_mean"), critic=SimpleNamespace(model=SimpleNamespace(path=None))),
        generator=SimpleNamespace(step_wise_trajectories=False)),
    dispatch=SimpleNamespace(get_lcm_dp_size=lambda: DP),
    all_metrics={},
)
out = RayPPOTrainer._balance_dp_tokens(fake, data)
orig = out["sequences"][:, 0].tolist()
assert sorted(orig) == list(range(N)), "not a permutation"
for s, e in [(0, 640), (640, 1280)]:
    assert all(s <= o < e for o in orig[s:e]), "sample left its mini-batch"
assert all(torch.equal(out["sequences"][j], seq[o]) and torch.equal(out["attention_mask"][j], am[o]) for j, o in enumerate(orig))
assert out.metadata["uids"] == [uids[o] for o in orig], "uids misaligned"
for s, e in [(0, 640), (640, 1280)]:
    k = (e - s) // DP
    loads = [sum(lens[o] for o in orig[s + r * k: s + (r + 1) * k]) for r in range(DP)]
    naive = [sum(lens[s + r * k: s + (r + 1) * k]) for r in range(DP)]
    print(f"mini-batch {s}-{e}: naive loads {naive} -> balanced {loads}")
    assert max(loads) - min(loads) <= max(lens), "not balanced"
print("metrics:", fake.all_metrics)
print("DP_BALANCE_TEST: PASS")
