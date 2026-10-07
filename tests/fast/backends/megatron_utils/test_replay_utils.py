from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import replay_utils
from miles.utils.replay_base import Replay


class _Replay(Replay):
    def record(self, top_indices):
        self.top_indices_list.append(top_indices)


class _Router(torch.nn.Module):
    def __init__(self, registered: list[Replay]):
        super().__init__()
        self.routing_replay = _Replay()
        registered.append(self.routing_replay)


class _Chunk(torch.nn.Module):
    def __init__(self, registered: list[Replay], num_layers: int):
        super().__init__()
        self.module = SimpleNamespace(config=SimpleNamespace(num_layers=num_layers, moe_layer_freq=1))
        self.decoder = torch.nn.ModuleList(_Router(registered) for _ in range(num_layers))


@pytest.fixture(autouse=True)
def _single_stage_layout(monkeypatch):
    monkeypatch.setattr(replay_utils, "get_num_layers_to_build", lambda config, vp_stage=None: config.num_layers)
    monkeypatch.setattr(replay_utils, "get_transformer_layer_offset", lambda config, vp_stage=None: 0)


def test_rollout_routing_goes_to_the_live_routers_of_a_rebuilt_decoder():
    """Qwen3VLGPTModel builds GPTModel's decoder, then replaces it: both sets of routers registered."""
    registered: list[Replay] = []
    chunk = _Chunk(registered, num_layers=3)
    discarded = list(registered)
    chunk.decoder = torch.nn.ModuleList(_Router(registered) for _ in range(3))
    replay_data = torch.arange(5 * 3 * 2).reshape(5, 3, 2)

    replay_utils.register_replay_list_moe(registered, replay_data, models=[chunk])

    for layer, router in enumerate(chunk.decoder):
        [recorded] = router.routing_replay.top_indices_list
        torch.testing.assert_close(recorded, replay_data[:, layer])
    assert all(not replay.top_indices_list for replay in discarded)


def test_router_count_that_does_not_match_the_moe_layers_is_rejected():
    registered: list[Replay] = []
    chunk = _Chunk(registered, num_layers=3)
    chunk.decoder.append(_Router(registered))

    with pytest.raises(AssertionError, match="4 routers with a replay"):
        replay_utils.register_replay_list_moe(registered, torch.zeros(5, 3, 2), models=[chunk])
