"""Record BeamNG.drive's *native* gameplay replay alongside the observations.

BeamNG.tech has a built-in replay system: the game engine keeps a single
"file stream" (``be:getFileStream()``) that can ``record`` the full simulation
into a ``.rpl`` file and later ``play`` it back from the in-game replay menu.
Unlike a video, a ``.rpl`` replay is the actual game state, so it can be
replayed, scrubbed and re-rendered from any camera - which makes it ideal for
a human to review and validate the drift runs we collect.

This module drives that system from Python over tcom (``queue_lua_command``),
so a replay can be started/stopped in lock-step with each collected episode.
The replay files land in the game's own ``replays/`` folder (separate from the
observation dataset) and the path of each file is returned so the caller can
associate it with the episode it belongs to.

The recorder is deliberately decoupled from the env: it only needs an object
exposing ``queue_lua_command(chunk, response=...)`` (a ``beamngpy`` connection),
so it can be tested against a stub.
"""
import time
from datetime import datetime
from typing import Optional

# The game writes replays under the user-data root; this mirrors the path the
# game's own replay module (lua/ge/extensions/core/replay.lua) uses.
REPLAY_DIR = "replays/"


class BeamNGReplayRecorder:
    """Start/stop the game's native replay recording per episode.

    Parameters
    ----------
    beamng:
        A ``beamngpy`` connection (anything with ``queue_lua_command``).
    prefix:
        Filename prefix for the ``.rpl`` files (e.g. ``"drift"``).
    """

    def __init__(self, beamng, prefix: str = "drift"):
        # ``beamng`` may be a connection object or a zero-arg callable that
        # returns one (the env's ``beamng`` only exists after the first reset,
        # so callers often pass ``lambda: env.unwrapped.beamng``).
        self._beamng = beamng
        self.prefix = prefix
        self._current: Optional[str] = None

    def _connection(self):
        return self._beamng() if callable(self._beamng) else self._beamng

    # -- lua helpers -------------------------------------------------------
    def _lua(self, chunk: str):
        return self._connection().queue_lua_command(chunk, response=True)

    @property
    def is_recording(self) -> bool:
        """True while this recorder has an active replay (Python-side tracking)."""
        return self._current is not None

    # -- public API --------------------------------------------------------
    def start(self, index: int) -> Optional[str]:
        """Begin recording a replay for episode ``index``.

        Returns the replay filename (relative to the game user-data root), or
        ``None`` if the engine has no file stream available.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{REPLAY_DIR}{self.prefix}_{timestamp}_{index:06d}.rpl"
        # stop any stale recording first so we always start from a clean state
        self._lua("local s = be:getFileStream(); if s then s:stop() end")
        result = self._lua(
            "local s = be:getFileStream()\n"
            "if not s then return 'no_stream' end\n"
            f'FS:directoryCreate("{REPLAY_DIR}", true)\n'
            f's:record("{filename}")\n'
            "return 'ok'"
        )
        if result == "ok":
            self._current = filename
            return filename
        print(f"[replay] start failed: {result!r}")
        self._current = None
        return None

    def stop(self, index: int) -> Optional[str]:
        """Stop the replay recording. Returns the filename that was recorded."""
        if self._current is None:
            return None
        self._lua("local s = be:getFileStream(); if s then s:stop() end")
        filename = self._current
        self._current = None
        return filename

    def play(self, filename: str, autoplay: bool = True) -> bool:
        """Play back a recorded replay in-game and confirm it started.

        ``filename`` is the game-relative path of a ``.rpl`` file (e.g.
        ``replays/drift_....rpl``). The game's native replay module
        (``core_replay.loadFile``) takes over the view and runs the replay, so
        a human can watch the drift from any camera. Returns True once the
        engine reports it is in playback.
        """
        autoplay_lua = "true" if autoplay else "false"
        self._lua(f'core_replay.loadFile("{filename}", {autoplay_lua})')
        # loadFile is synchronous, but confirm with a short poll in case the
        # engine needs a frame to enter playback
        state = "inactive"
        for _ in range(10):
            state = self._lua("return core_replay.getState()")
            if state == "playback":
                return True
            time.sleep(0.2)
        print(f"[replay] play failed for {filename!r}: state={state!r}", flush=True)
        return False
