"""Board → profile binding for Kanban control-plane aux calls.

A board names its owning profile in ``board.json`` (``owner_profile``). Triage
specify/decompose then run with that profile's HERMES_HOME, so
``auxiliary.triage_specifier`` / ``auxiliary.kanban_decomposer``, ``auth.json``
and the routing fallbacks come from the board's profile instead of whichever
profile happens to host the dispatcher.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_boards as boards
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as specify
from hermes_constants import get_hermes_home


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return h


def _work_profile(home: Path) -> Path:
    work = home / "profiles" / "work"
    work.mkdir(parents=True)
    (work / "config.yaml").write_text(
        "auxiliary:\n"
        "  kanban_decomposer:\n"
        "    provider: anthropic\n"
        "    model: work-decomposer\n"
        "  triage_specifier:\n"
        "    provider: anthropic\n"
        "    model: work-specifier\n"
        "kanban:\n"
        "  orchestrator_profile: work\n"
        "  default_assignee: work\n",
        encoding="utf-8",
    )
    return work


def _board(home: Path, *, owner: str = "work", slug: str = "work") -> None:
    bdir = home / "kanban" / "boards" / slug
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "board.json").write_text(
        json.dumps({"slug": slug, "name": "Work", "owner_profile": owner}),
        encoding="utf-8",
    )


def _reply(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def test_owner_profile_reads_binding_and_ignores_unknown(home):
    _work_profile(home)
    _board(home)
    assert boards.board_owner_profile("work") == "work"
    assert boards.board_owner_profile("unbound") is None
    _board(home, owner="ghost", slug="ghost-board")
    assert boards.board_owner_profile("ghost-board") is None


def test_scoped_board_profile_switches_config_and_restores(home):
    work = _work_profile(home)
    _board(home)
    from hermes_cli.config import load_config_readonly

    before = str(get_hermes_home())
    with boards.scoped_board_profile("work") as name:
        assert name == "work"
        assert str(get_hermes_home()) == str(work)
        cfg = load_config_readonly()
        assert cfg["auxiliary"]["kanban_decomposer"]["model"] == "work-decomposer"
    assert str(get_hermes_home()) == before


def test_specify_runs_aux_with_owner_model(home, monkeypatch):
    work = _work_profile(home)
    _board(home)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "work")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough idea", triage=True)

    seen: dict[str, Any] = {}

    def fake_call_llm(*, task, **kwargs):
        from agent.auxiliary_task_config import _get_auxiliary_task_config

        seen["model"] = _get_auxiliary_task_config(task).get("model")
        seen["home"] = str(get_hermes_home())
        return _reply('{"title": "tight title", "body": "concrete body"}')

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    outcome = specify.specify_task(tid)
    assert outcome.ok, outcome.reason
    assert seen["model"] == "work-specifier"
    assert seen["home"] == str(work)


def test_decompose_uses_owner_routing_and_model(home, monkeypatch):
    _work_profile(home)
    _board(home)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "work")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship it", triage=True)

    seen: dict[str, Any] = {}

    def fake_call_llm(*, task, **kwargs):
        from agent.auxiliary_task_config import _get_auxiliary_task_config

        seen["model"] = _get_auxiliary_task_config(task).get("model")
        return _reply(json.dumps({
            "fanout": True,
            "rationale": "one leaf",
            "tasks": [{"title": "do it", "body": "spec", "assignee": None, "parents": []}],
        }))

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    outcome = decomp.decompose_task(tid)
    assert outcome.ok, outcome.reason
    assert seen["model"] == "work-decomposer"
    assert outcome.child_ids
    with kbc.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    # Without the board binding this falls back to the active profile ("default").
    assert child is not None and child.assignee == "work"
