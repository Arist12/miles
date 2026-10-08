from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from miles.backends.training_utils.weight_update.protocols import broadcast


@pytest.fixture
def calls(monkeypatch):
    log = {"connect": [], "disconnect": [], "update": []}
    monkeypatch.setattr(broadcast, "get_data_replica_rank_and_size", lambda *_: (0, 1))
    monkeypatch.setattr(
        broadcast,
        "connect_rollout_engines_from_distributed",
        lambda args, name, engines, engine_gpu_counts: log["connect"].append((name, list(engines), engine_gpu_counts))
        or f"group-{name}",
    )
    monkeypatch.setattr(
        broadcast,
        "disconnect_rollout_engines_from_distributed",
        lambda args, name, group, engines: log["disconnect"].append((name, group)),
    )
    monkeypatch.setattr(
        broadcast,
        "update_weights_from_distributed",
        lambda name, group, engines, bucket, selector, transfer_mode: log["update"].append((name, group, list(engines)))
        or [],
    )
    return log


def _protocol(engines_per_group: int):
    protocol = object.__new__(broadcast.UpdateWeightFromDistributed)
    protocol.args = SimpleNamespace(
        update_weight_engines_per_group=engines_per_group,
        rollout_num_gpus_per_engine=2,
        update_weight_transfer_mode="broadcast",
    )
    protocol._engine_groups = []
    protocol._engine_lock = nullcontext()
    return protocol


def _connect(protocol, engines):
    protocol.connect(engines, None, None, SimpleNamespace(), SimpleNamespace(gather_pp=True), "all")


def test_default_puts_every_engine_in_one_group(calls):
    protocol = _protocol(0)
    _connect(protocol, ["e0", "e1", "e2"])
    protocol.send_bucket([("w", object())])

    assert calls["connect"] == [("miles-pp_0", ["e0", "e1", "e2"], [2, 2, 2])]
    assert calls["update"] == [("miles-pp_0", "group-miles-pp_0", ["e0", "e1", "e2"])]


def test_engines_are_split_into_groups_and_each_group_gets_the_bucket(calls):
    protocol = _protocol(2)
    _connect(protocol, ["e0", "e1", "e2"])
    bucket = [("w", object())]
    protocol.send_bucket(bucket)

    assert calls["connect"] == [
        ("miles-pp_0_e0", ["e0", "e1"], [2, 2]),
        ("miles-pp_0_e2", ["e2"], [2]),
    ]
    assert calls["update"] == [
        ("miles-pp_0_e0", "group-miles-pp_0_e0", ["e0", "e1"]),
        ("miles-pp_0_e2", "group-miles-pp_0_e2", ["e2"]),
    ]
    assert bucket == []


def test_reconnecting_tears_down_every_previous_group(calls):
    protocol = _protocol(1)
    _connect(protocol, ["e0", "e1"])
    _connect(protocol, ["e0", "e1"])

    assert calls["disconnect"] == [
        ("miles-pp_0_e0", "group-miles-pp_0_e0"),
        ("miles-pp_0_e1", "group-miles-pp_0_e1"),
    ]
