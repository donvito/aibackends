from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest

from aibackends.backends.routing import get_routing_backend, list_routing_backends
from aibackends.core.exceptions import TaskExecutionError
from aibackends.tasks import route_prompt, route_prompts
from aibackends.tasks.registry import get_task

router_module = importlib.import_module("aibackends.backends.routing.lfm2_router")


class _FakeRouterModel:
    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict[str, Any]] = []

    def route(
        self,
        text: str,
        routes: list[str],
        *,
        tokenizer: Any,
        threshold: float | None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "text": text,
                "routes": list(routes),
                "tokenizer": tokenizer,
                "threshold": threshold,
            }
        )
        return [dict(item) for item in self.results]


@pytest.fixture(autouse=True)
def _reset_router_cache() -> Iterator[None]:
    router_module.clear_model_cache()
    yield
    router_module.clear_model_cache()


def _install_fake_model(
    fake: _FakeRouterModel,
    *,
    device: str = "cpu",
) -> _FakeRouterModel:
    cache_key = (router_module.ROUTER_MODEL_ID, device)
    router_module._MODEL_CACHE[cache_key] = (object(), fake)
    return fake


def test_route_prompt_ranks_routes_and_picks_best() -> None:
    fake = _install_fake_model(
        _FakeRouterModel(
            results=[
                {"route": "sales", "score": 0.2},
                {"route": "coding", "score": 0.7},
                {"route": "billing", "score": 0.1},
            ]
        )
    )

    result = route_prompt(
        "Can you help me debug a failing Python unit test?",
        ["coding", "sales", "billing"],
    )

    assert result.best_route == "coding"
    assert [score.route for score in result.scores] == ["coding", "sales", "billing"]
    assert [score.score for score in result.scores] == [0.7, 0.2, 0.1]
    assert result.backend_used == "lfm2-prompt-router"
    assert result.model_id == router_module.ROUTER_MODEL_ID
    assert result.text == "Can you help me debug a failing Python unit test?"
    assert fake.calls[0]["routes"] == ["coding", "sales", "billing"]
    assert fake.calls[0]["threshold"] is None


def test_route_prompt_passes_threshold_and_strips_routes() -> None:
    fake = _install_fake_model(
        _FakeRouterModel(results=[{"route": "coding", "score": 0.9}])
    )

    result = route_prompt("Fix my SQL query.", ["  coding ", "sales"], threshold=0.4)

    assert result.best_route == "coding"
    assert fake.calls[0]["routes"] == ["coding", "sales"]
    assert fake.calls[0]["threshold"] == 0.4


def test_route_prompt_returns_no_best_route_when_all_filtered() -> None:
    _install_fake_model(_FakeRouterModel(results=[]))

    result = route_prompt("Completely off-topic text.", ["coding"], threshold=0.9)

    assert result.best_route is None
    assert result.scores == []


def test_route_prompt_rejects_invalid_model_output() -> None:
    _install_fake_model(_FakeRouterModel(results=[{"route": 1, "score": "high"}]))

    with pytest.raises(TaskExecutionError, match="invalid route entry"):
        route_prompt("hello", ["coding"])


def test_route_prompts_routes_each_text() -> None:
    fake = _install_fake_model(
        _FakeRouterModel(results=[{"route": "coding", "score": 0.8}])
    )

    results = route_prompts(["Fix this bug.", "Write a poem."], ["coding", "creative"])

    assert len(results) == 2
    assert [call["text"] for call in fake.calls] == ["Fix this bug.", "Write a poem."]
    assert all(result.best_route == "coding" for result in results)


def test_route_prompts_returns_empty_for_no_inputs() -> None:
    _install_fake_model(_FakeRouterModel())

    assert route_prompts([], ["coding"]) == []


def test_router_backend_validates_options() -> None:
    backend = get_routing_backend("prompt-router")

    with pytest.raises(ValueError, match="at least one label"):
        backend.route("hello", [])
    with pytest.raises(ValueError, match="non-empty strings"):
        backend.route("hello", ["coding", "  "])
    with pytest.raises(ValueError, match="threshold"):
        backend.route("hello", ["coding"], threshold=1.5)
    with pytest.raises(ValueError, match="Unsupported prompt router device"):
        backend.load(device="tpu")


def test_route_prompt_rejects_unknown_backend() -> None:
    with pytest.raises(TaskExecutionError, match="Unsupported routing backend"):
        route_prompt("hello", ["coding"], backend="unknown")


def test_routing_backend_alias_lookup() -> None:
    canonical = get_routing_backend("lfm2-prompt-router")

    assert get_routing_backend("lfm2.5-prompt-router") is canonical
    assert get_routing_backend("prompt-router") is canonical
    assert list_routing_backends() == ["lfm2-prompt-router"]


def test_load_router_model_normalizes_gpu_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[str] = []
    devices: list[str] = []

    class _FakeLoadedModel:
        def to(self, device: str) -> _FakeLoadedModel:
            devices.append(device)
            return self

        def eval(self) -> None:
            return None

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, *, trust_remote_code: bool) -> _FakeLoadedModel:
            loads.append(model_id)
            return _FakeLoadedModel()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, *, trust_remote_code: bool) -> object:
            return object()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = _AutoModel  # type: ignore[attr-defined]
    fake_transformers.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))

    first = router_module.load_router_model("gpu")
    second = router_module.load_router_model("cuda")

    assert first is second
    assert loads == [router_module.ROUTER_MODEL_ID]
    assert devices == ["cuda"]


def test_route_task_runs_with_cli_style_labels() -> None:
    _install_fake_model(
        _FakeRouterModel(results=[{"route": "device assistant", "score": 0.9}])
    )
    spec = get_task("route_prompt")

    assert spec.name == "route-prompt"
    assert spec.accepts_backend
    assert spec.accepts_labels
    assert spec.requires_labels
    assert spec.accepts_device
    assert spec.accepts_threshold
    assert not spec.accepts_runtime
    assert not spec.accepts_model

    result = spec.create().run("Set a timer for 10 minutes.", labels=["device assistant"])

    assert result.best_route == "device assistant"


def test_route_task_requires_routes() -> None:
    spec = get_task("route-prompt")

    with pytest.raises(ValueError, match="at least one route"):
        spec.create().run("hello")
