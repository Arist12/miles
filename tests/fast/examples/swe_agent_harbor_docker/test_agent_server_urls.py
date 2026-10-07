"""A multi-host run lists one agent server per host; each trial goes to the least busy one."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
AGENT_SCRIPT = REPO_ROOT / "examples" / "swe-agent-harbor-docker" / "swe_agent_function.py"


@pytest.fixture
def agent_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swe_agent_harbor_docker_agent", AGENT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comma_separated_servers_are_parsed(agent_module, monkeypatch):
    monkeypatch.setenv("AGENT_SERVER_URL", "http://a:11000/, http://b:11000")

    assert agent_module._agent_server_urls() == ["http://a:11000", "http://b:11000"]


def test_single_server_and_default(agent_module, monkeypatch):
    monkeypatch.delenv("AGENT_SERVER_URL", raising=False)
    monkeypatch.delenv("SWE_AGENT_URL", raising=False)

    assert agent_module._agent_server_urls() == ["http://localhost:11000"]


def test_trials_go_to_the_server_with_fewest_in_flight(agent_module):
    urls = ["http://a:11000", "http://b:11000", "http://c:11000"]
    agent_module._in_flight.update({"http://a:11000": 3, "http://b:11000": 1, "http://c:11000": 2})

    assert agent_module._pick_agent_server(urls) == "http://b:11000"


def test_unused_servers_count_as_idle(agent_module):
    agent_module._in_flight.update({"http://a:11000": 1})

    assert agent_module._pick_agent_server(["http://a:11000", "http://b:11000"]) == "http://b:11000"
