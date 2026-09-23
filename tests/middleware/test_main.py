"""Actionable diagnostics for missing runtime dependencies."""

import builtins
import logging
import sys

from middleware.main import main


def test_missing_dependency_names_module_and_interpreter(monkeypatch, caplog):
    """A missing dotenv dependency must be diagnosed before reading any CSV."""
    original = builtins.__import__

    def missing_dotenv(name, *args, **kwargs):
        """Simulate an interpreter without the dotenv distribution."""
        if name == "dotenv":
            raise ModuleNotFoundError("No module named dotenv", name="dotenv")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_dotenv)
    with caplog.at_level(logging.ERROR):
        assert main(["--csv", "unused.csv", "--dry-run"]) == 1
    assert "dotenv" in caplog.text
    assert sys.executable in caplog.text
    assert "pip install -r" in caplog.text
