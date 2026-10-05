"""All-gather the MoE alltoall dispatcher's per-expert token counts over Gloo (``--moe-token-counts-over-gloo``).

The dispatcher derives every rank's all-to-all output splits from an all-gather of these counts.
Observed with RCCL 2.27.7 (ROCm 7.2.4) across 4 nodes: about once per 20 rollouts one rank's
output splits disagreed with what its peers sent it, and that rank then waited in the all-to-all
for rows that never came. Gloo moves the same few integers on the host, at the cost of one
device-to-host copy per MoE layer.
"""

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_gloo_groups: dict[tuple[int, ...], dist.ProcessGroup] = {}


def _gloo_twin(group: dist.ProcessGroup) -> dist.ProcessGroup:
    """A Gloo group with `group`'s ranks, in its order, created by its members on first use."""
    ranks = tuple(dist.get_process_group_ranks(group))
    if ranks not in _gloo_groups:
        _gloo_groups[ranks] = dist.new_group(list(ranks), backend="gloo", use_local_synchronization=True)
    return _gloo_groups[ranks]


def gather_token_counts(counts: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """``counts`` of every rank of ``group``, concatenated in group order, on ``counts``' device."""
    cpu = counts.detach().cpu()
    gathered = [torch.empty_like(cpu) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, cpu, group=_gloo_twin(group))
    return torch.cat(gathered).to(counts.device)


def install() -> None:
    """Route the alltoall dispatcher's token-count gather (its only int64 gather) through Gloo."""
    from megatron.core.tensor_parallel import mappings
    from megatron.core.transformer.moe import token_dispatcher

    gather = token_dispatcher.gather_from_sequence_parallel_region
    assert gather is mappings.gather_from_sequence_parallel_region, "the MoE dispatcher's count gather moved"

    def gather_from_sequence_parallel_region(input_, *args, group=None, **kwargs):
        if input_.dtype != torch.int64 or input_.dim() != 1 or group is None or args or kwargs:
            return gather(input_, *args, group=group, **kwargs)
        return gather_token_counts(input_, group)

    token_dispatcher.gather_from_sequence_parallel_region = gather_from_sequence_parallel_region
    logger.info("MoE token counts are all-gathered over Gloo")
