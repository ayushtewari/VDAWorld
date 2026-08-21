import numpy as np


class SimulatorBase:
    def __init__(self, frame_size=(1024, 576), api=None, fps=30):
        # Define state variables
        self.t = 0

        # Provided parameters
        self.frame_size = frame_size
        self.api = api
        self.fps = fps

        self._debug_view: np.ndarray | None = None

    def log_debug_view(self, image: np.ndarray) -> None:
        """Store a debug image retrievable via the view_debug tool.

        Call this from fit(), render_frame(), or update_simulation() to expose
        an auxiliary view — e.g. a component-only render, annotated overlay, or
        intermediate state — to help diagnose something confusing or in need of
        refinement. Only the last image passed is retained.

        Arguments:
            image (np.ndarray[np.uint8])
                RGB image of shape (H, W, 3) and dtype uint8.
        """
        self._debug_view = image

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
                The input the simulation is fitted to. In image mode this is a
                single frame of shape [H, W, 3] (height, width, RGB) that the
                simulation's first frame must match. In video mode this is the
                full input video of shape [N, H, W, 3]; the simulation
                *continues* the video, so fit the initial state to the last
                frame, image[-1].
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the simulation to its initial state.
        """
        self.t = 0

    def __next__(self):
        """
        Return the next frame and advance by one timestep.

        Calls render_frame() for the current state, then advances via
        update_simulation(dt=1/fps).

        Returns:
            frame (np.ndarray[np.uint8])
                Rendered simulation frame of shape (height, width, 3).
        """
        frame = self.render_frame()
        self.update_simulation(dt=1 / self.fps)
        self.t += 1 / self.fps

        target_height = self.frame_size[1]
        target_width = self.frame_size[0]
        height = frame.shape[0]
        width = frame.shape[1]
        assert (height, width) == (
            target_height,
            target_width,
        ), f"Expected frame size {target_height}x{target_width}, got {height}x{width}"
        if len(frame.shape) == 3:
            assert frame.shape[2] == 3, (
                "Expected frame to have 3 color channels as it has 3 dimensions"
            )
        else:
            assert len(frame.shape) == 2, (
                f"Expected frame to have 2 dimensions (height, width) or 3 dimensions (height, width, channels). Got {len(frame.shape)} dimensions instead."
            )
        assert frame.dtype == np.uint8

        return frame

    def run_simulation(self, n_frames, out_arr=None):
        frames = []
        for i in range(n_frames):
            frame = next(self)
            if out_arr is not None:
                np.copyto(out_arr[i], frame)
            else:
                frames.append(frame)
        return out_arr if out_arr is not None else frames

    def __iter__(self):
        return self

    def __del__(self):
        pass
