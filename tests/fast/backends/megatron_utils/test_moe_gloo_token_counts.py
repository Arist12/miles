"""The Gloo token-count gather must match the device gather, in group order, and touch nothing else."""

import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils import moe_gloo_token_counts


def _worker_subgroup(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        members = [1, 3]
        group = dist.new_group(ranks=members)
        if rank in members:
            # Only the members take part, as only they run that group's dispatcher.
            counts = torch.arange(3, dtype=torch.int64) + 10 * rank
            gathered = moe_gloo_token_counts.gather_token_counts(counts, group)
            assert torch.equal(gathered, torch.cat([torch.arange(3, dtype=torch.int64) + 10 * r for r in members]))
    finally:
        dist.destroy_process_group()


def test_gathers_a_subgroup_in_group_order():
    run_multiprocess(_worker_subgroup, world_size=4)


def test_install_routes_only_the_int64_counts(monkeypatch):
    from megatron.core.tensor_parallel import mappings
    from megatron.core.transformer.moe import token_dispatcher

    calls = []

    def device_gather(x, *args, group=None, **kwargs):
        calls.append("device")

    monkeypatch.setattr(mappings, "gather_from_sequence_parallel_region", device_gather)
    monkeypatch.setattr(token_dispatcher, "gather_from_sequence_parallel_region", device_gather)
    monkeypatch.setattr(moe_gloo_token_counts, "gather_token_counts", lambda x, group: calls.append("gloo"))
    moe_gloo_token_counts.install()
    gather = token_dispatcher.gather_from_sequence_parallel_region
    group = object()

    gather(torch.ones(2), group=group)  # hidden states / probs
    gather(torch.ones(2, dtype=torch.int64), group=group)  # token counts
    gather(torch.ones(2, dtype=torch.int64))  # no explicit group
    gather(torch.ones(2, dtype=torch.int64), group=group, output_split_sizes=[1, 1])

    assert calls == ["device", "gloo", "device", "device"]
