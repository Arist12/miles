"""The P2P ring must reproduce all_to_all_single exactly, including on changing uneven splits."""

import random

import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils.ep_p2p_alltoall import ring_all_to_all_single

HIDDEN = 3


def _split_matrix(world_size: int, seed: int, *, allow_zero: bool) -> list[list[int]]:
    """counts[src][dst] in group order, identical on every rank because the seed is shared."""
    rng = random.Random(seed)
    low = 0 if allow_zero else 1
    return [[rng.randint(low, 7) for _ in range(world_size)] for _ in range(world_size)]


def _payload(src: int, dst: int, count: int) -> torch.Tensor:
    """Rows that name their sender, receiver and position, so misplacement is visible."""
    base = torch.arange(count, dtype=torch.float32).unsqueeze(1).expand(count, HIDDEN)
    return base + 1000.0 * src + 100.0 * dst


def _exchange(counts: list[list[int]], group) -> None:
    me = dist.get_rank(group)
    world = dist.get_world_size(group)
    input_splits = counts[me]
    output_splits = [counts[src][me] for src in range(world)]
    send = torch.cat([_payload(me, dst, n) for dst, n in enumerate(input_splits)])
    recv = torch.full((sum(output_splits), HIDDEN), -1.0)

    ring_all_to_all_single(recv, send, output_splits, input_splits, group)

    expected = torch.cat([_payload(src, me, n) for src, n in enumerate(output_splits)])
    torch.testing.assert_close(recv, expected, rtol=0, atol=0)


def _worker_changing_splits(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        for iteration in range(20):
            _exchange(_split_matrix(world_size, seed=iteration, allow_zero=iteration % 2 == 0), group=None)
    finally:
        dist.destroy_process_group()


def _worker_subgroup(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        # Group order differs from global order, so peers must be addressed by global rank.
        group = dist.new_group(ranks=[2, 0, 3, 1])
        for iteration in range(5):
            _exchange(_split_matrix(world_size, seed=100 + iteration, allow_zero=True), group)
    finally:
        dist.destroy_process_group()


def test_ring_matches_all_to_all_on_changing_uneven_splits():
    run_multiprocess(_worker_changing_splits, world_size=4)


def test_ring_addresses_peers_by_global_rank_in_a_reordered_group():
    run_multiprocess(_worker_subgroup, world_size=4)
