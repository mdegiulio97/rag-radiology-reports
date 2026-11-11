"""Smoke test of the Streamlit app (offline): renders without Ollama and without exceptions."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "main.py")


def test_app_renders_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception


def test_app_answers_in_corpus_query():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    # set an in-corpus question (in Italian, like the corpus) and click "Ask"
    at.text_input[0].set_value("Ci sono segni di polmonite?")
    at.button[0].click().run()
    assert not at.exception
