```python
import numpy as np

class SimulatorBase:
    def __init__(self, frame_size=(1024, 576), api=None, fps=30):
        # Define state variables
        self.t = 0

        # Provided parameters
        self.frame_size = frame_size
        self.api = api
        self.fps = fps

    def render_frame(self):
        """
        Render the next frame of the simulation.
        """
        raise NotImplementedError()

    def update_simulation(self, dt: float):
        """
        Update the simulation by one timestep dt.

        Arguments:
            dt (float)
                Time step to update the simulation by.
        """
        raise NotImplementedError()

    def fit(self, image: np.ndarray):
        """
        Fit the parameters of the scene to the provided input.

        Arguments:
            image (np.ndarray[np.uint8])
                [FIT_INPUT_DESC]
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the simulation to its initial state.
        """
        self.t = 0

    def log_debug_view(self, image: np.ndarray) -> None:
        """
        Store a debug image retrievable via the view_debug tool.

        Call from fit(), render_frame(), or update_simulation() to expose
        an auxiliary view — e.g. a component-only render, annotated overlay,
        or intermediate state — when something is confusing or needs refinement.
        Only the last image passed is retained.

        Arguments:
            image (np.ndarray[np.uint8])
                RGB image of shape (H, W, 3) and dtype uint8.
        """

    def __next__(self):
        """
        Return the next frame and advance by one timestep.

        Calls render_frame() for the current state, then advances via
        update_simulation(dt=1/fps).

        Returns:
            frame (np.ndarray[np.uint8]) of shape (height, width, 3).
        """
        # Calls render_frame(), then update_simulation(dt=1/fps).
        # Asserts frame matches self.frame_size and dtype is np.uint8.

    def run_simulation(self, n_frames):
        # Returns frames — a list of n_frames arrays from __next__.
        pass
```
