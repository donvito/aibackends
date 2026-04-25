from __future__ import annotations

from typer.testing import CliRunner

from aibackends.cli import app

runner = CliRunner()


def test_task_command_accepts_runtime_and_model_strings():
    result = runner.invoke(
        app,
        [
            "task",
            "summarize",
            "--input",
            "Meeting notes",
            "--runtime",
            "stub",
            "--model",
            "stub-model",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "Stub summary"


def test_check_command_accepts_runtime_and_model_strings():
    result = runner.invoke(
        app,
        [
            "check",
            "stub",
            "--model",
            "stub-model",
        ],
    )

    assert result.exit_code == 0
    assert '"runtime": "stub"' in result.stdout
    assert '"client": "StubRuntime"' in result.stdout
    assert '"model": "stub-model"' in result.stdout
