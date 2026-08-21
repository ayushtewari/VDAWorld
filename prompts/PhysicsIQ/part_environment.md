The simulator must be written in Python as a class `[SIMULATOR_CLASS_NAME]` which inherits the `SimulatorBase` base class. The specification of this `SimulatorBase` class is provided below. The `[SIMULATOR_CLASS_NAME]` class overrides the methods of this base class: `fit`, which fits the simulator parameters to the image; `update_simulation`, which steps the simulation forward by a specified timestep; and `render_frame`, which renders a single frame of the simulation. Additionally, the `reset` method can be overridden to reset the simulation to its initial state, and the `__init__` method. You may also add any helper methods or classes as needed.

[SPEC_SIMULATOR]

As well as standard python libraries, the libraries `numpy`, `scipy`, `scikit-image`, `trimesh`, `mujoco`, `pygame`, and `pymunk` are available.

Recall if using `mujoco` that when specifying video dimensions, your XML should also include:

```xml
<visual>
  <global offwidth="my_width"/>
</visual>
```

Additionally, if using complex simulations such as particles or soft bodies, consider using a `CG` or `PCG` solver in `mujoco` for speed and stability e.g.:

```xml
<option solver="CG" iterations="50"/>
```

For soft fabric simulations in `mujoco`, do not use the inbuilt `mujoco` cloth implementations.

When using pygame or pymunk, ensure that the rendering is done off-screen to avoid issues with headless environments. You can use `pygame.Surface` for off-screen rendering and then convert it to an image format suitable for output.
