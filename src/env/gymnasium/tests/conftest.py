"""Shared pytest configuration and fixtures for the BeamNG tech env tests.

All tests live in this folder. The pure-math tests (``test_drift_math.py``)
run without the game. The env and benchmark tests (marked ``game``) share a
single BeamNG.drive instance for the whole session: the game is launched once
on the first ``env.reset()`` (~48 s) and every test calls ``env.reset()`` for
a clean state.

Run only the fast tests (no game):
    pytest -m "not game"
Run everything (needs BNG_HOME, DISPLAY and XAUTHORITY set):
    pytest
Run the interactive render test (blocks until Ctrl+C):
    pytest -m interactive --interactive
"""
import os
import subprocess
import sys
import time

import pytest

# Make the package importable no matter where pytest is invoked from.
_GYM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _GYM_DIR not in sys.path:
    sys.path.insert(0, _GYM_DIR)

# Importing the package registers the env id used by gym.make below.
import beamng_tech_env  # noqa: F401


def pytest_addoption(parser):
    parser.addoption(
        "--interactive",
        action="store_true",
        default=False,
        help="run interactive tests (e.g. render, which blocks until Ctrl+C)",
    )


def pytest_collection_modifyitems(config, items):
    # Skip interactive tests unless --interactive was passed. Done here (rather
    # than via addopts -m) so that a command-line -m selection still works.
    if config.getoption("--interactive", default=False):
        return
    skip_interactive = pytest.mark.skip(reason="interactive test (pass --interactive to run)")
    for item in items:
        if "interactive" in item.keywords:
            item.add_marker(skip_interactive)


def _game_available():
    """Return (ok, reason): the game tests need the game home and an X display."""
    bng_home = os.environ.get("BNG_HOME", "")
    if not bng_home or not os.path.isdir(bng_home):
        return False, "BNG_HOME is not set or does not exist"
    if not os.environ.get("DISPLAY", ""):
        return False, "DISPLAY is not set"
    if not os.environ.get("XAUTHORITY", ""):
        return False, "XAUTHORITY is not set"
    return True, ""


def _kill_game():
    """Kill a leftover game process so the session starts from a clean slate."""
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "BeamNG.tech.x64"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass
    time.sleep(1)


@pytest.fixture(scope="session")
def env():
    """One shared environment (and game) for the whole test session.

    The game is launched lazily on the first reset(); each test is expected to
    call ``env.reset()`` for a clean state. Any leftover game is killed first
    so an instance from a previous run cannot interfere.
    """
    ok, reason = _game_available()
    if not ok:
        pytest.skip(f"game tests skipped: {reason}")
    _kill_game()
    import gymnasium as gym

    environment = gym.make("beamng_tech_env/BeamNGTech-v0.39", render_mode=None)
    yield environment
    environment.close()
