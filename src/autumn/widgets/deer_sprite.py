"""Small looping animated sprite: a deer running right, used under the
"AUTUMN" title on the landing screen (InputScreen). Frame data is baked into
`_deer_frames.py` -- see that module's docstring for provenance."""

from textual.widgets import Static

from autumn.widgets._deer_frames import DEER_RUN_RIGHT_FRAMES

_FRAME_INTERVAL_SECONDS = 0.12


class DeerSprite(Static):
    """Cycles through `DEER_RUN_RIGHT_FRAMES` on a timer, looping forever."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(DEER_RUN_RIGHT_FRAMES[0], *args, **kwargs)
        self._frame_index = 0

    def on_mount(self) -> None:
        self.set_interval(_FRAME_INTERVAL_SECONDS, self._advance_frame)

    def _advance_frame(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(DEER_RUN_RIGHT_FRAMES)
        self.update(DEER_RUN_RIGHT_FRAMES[self._frame_index])
