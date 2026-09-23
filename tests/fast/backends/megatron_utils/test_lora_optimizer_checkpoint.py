"""LoRA checkpoint coverage for distributed optimizer parameter state."""

import sys
import types
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer

import miles.backends.megatron_utils.lora.utils as lora_utils


class _Child(DistributedOptimizer):
    def __init__(self, *, stub=False, dp_rank=0):
        self.is_stub_optimizer = stub
        self.data_parallel_group = SimpleNamespace(rank=lambda: dp_rank)
        self.optimizer = None if stub else SimpleNamespace(param_groups=[])
        self.training_state = "not called"
        self.loaded = "not called"

    def state_dict(self):
        assert not self.is_stub_optimizer
        return {"step": 3}

    def load_state_dict(self, state_dict):
        assert not self.is_stub_optimizer
        self.training_state = state_dict

    def save_parameter_state(self, filename):
        if self.data_parallel_group.rank() == 0:
            torch.save({"master": 1}, filename)

    def load_parameter_state(self, filename):
        self.loaded = torch.load(filename) if self.data_parallel_group.rank() == 0 else None


def _chain(*children):
    chain = ChainedOptimizer.__new__(ChainedOptimizer)
    chain.chained_optimizers = list(children)
    return chain


def _rank0_parallel_state():
    rank0 = SimpleNamespace(rank=0)
    return SimpleNamespace(effective_dp=rank0, cp=rank0, tp=rank0, pp=rank0)


def test_round_trip_skips_stub_children_and_reads_through_the_data_parallel_root(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_utils, "get_parallel_state", _rank0_parallel_state)
    # the HF PEFT export is best-effort and needs a real bridge; fail it fast
    bridge = types.ModuleType("megatron.bridge")
    bridge.AutoBridge = SimpleNamespace(from_hf_pretrained=Mock(side_effect=RuntimeError("no bridge in this test")))
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge)
    args = Namespace(hf_checkpoint="/nonexistent", megatron_to_hf_mode="bridge", no_save_optim=False)
    scheduler = SimpleNamespace(state_dict=lambda: {"num_steps": 8})
    optimizer = _chain(_Child(dp_rank=0), _Child(stub=True), _Child(dp_rank=1))
    lora_utils.save_lora_checkpoint(
        [], args, str(tmp_path), optimizer=optimizer, opt_param_scheduler=scheduler, iteration=3
    )
    assert [p.name for p in tmp_path.glob("optimizer_param_state*")] == ["optimizer_param_state_rank0_optimizer0.pt"]

    root, stub, peer = _Child(dp_rank=0), _Child(stub=True), _Child(dp_rank=1)
    scheduler = MagicMock()
    assert lora_utils._load_training_state(tmp_path, _chain(root, stub, peer), scheduler) == 3

    assert (root.training_state, peer.training_state) == ({"step": 3}, {"step": 3})
    assert (root.loaded, peer.loaded, stub.loaded) == ({"master": 1}, None, "not called")
    scheduler.load_state_dict.assert_called_once_with({"num_steps": 8})


def test_checkpoint_without_parameter_state_keeps_the_fresh_optimizer(tmp_path):
    child = _Child(dp_rank=0)
    torch.save({"iteration": 3, "optimizer": [{"step": 3}]}, tmp_path / "training_state_rank0.pt")

    assert lora_utils._load_training_state(tmp_path, _chain(child), None) == 3
    assert (child.training_state, child.loaded) == ("not called", "not called")


def test_partial_parameter_state_is_rejected(tmp_path):
    optimizer = _chain(_Child(dp_rank=0), _Child(dp_rank=0))
    torch.save({"iteration": 3, "optimizer": [{"step": 3}, {"step": 3}]}, tmp_path / "training_state_rank0.pt")
    (tmp_path / "optimizer_param_state_rank0_optimizer1.pt").touch()

    with pytest.raises(RuntimeError, match="Optimizer parameter state is incomplete"):
        lora_utils._load_training_state(tmp_path, optimizer, None)


def test_masters_are_refreshed_whenever_adapter_weights_are_written(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_utils, "get_parallel_state", _rank0_parallel_state)
    param = torch.nn.Parameter(torch.zeros(2))
    model = [SimpleNamespace(named_parameters=lambda: iter([("adapter.lora_A.weight", param)]))]
    torch.save({"adapter.lora_A.weight": torch.ones(2)}, tmp_path / "adapter_megatron_rank0.pt")
    optimizer = MagicMock(chained_optimizers=[_Child(dp_rank=0)])

    loaded, iteration = lora_utils.load_lora_adapter(model, str(tmp_path), optimizer=optimizer)

    assert (loaded, iteration) == (True, None)
    optimizer.reload_model_params.assert_called_once_with()
