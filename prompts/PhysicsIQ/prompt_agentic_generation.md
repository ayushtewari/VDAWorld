[PART_TASK]

[PART_ENVIRONMENT]

You have a maximum of [MAX_TURNS] turns to write, test, and refine your simulator iteratively. If you are satisfied the simulation is correct before using all turns, stop early by providing your final response without calling any tools.

## Available Tools

[SANDBOX_TOOLS]

## Strategy

Approach the implementation iteratively, using the available tools to verify your work at each stage. Use `run_simulation()` to test behaviour and the inspection tools listed above to diagnose issues. How you structure your work is up to you.

## API: self.api

[PART_TOOLBOX]

Your code must:

- Define only the class `[SIMULATOR_CLASS_NAME]` (and any helper classes or functions it needs).
- Assume that `SimulatorBase` and `WorldAPI` are already imported — do not import them yourself.
- Inherit from `SimulatorBase` and implement all its abstract methods: `fit`, `render_frame`, `update_simulation`.
- Access the WorldAPI via `self.api.<method>()`.

## Response

Once you have finished, respond with an honest assessment of the simulation's state. Do not include any code in your final response.

- **If you know the simulation is incorrect** (e.g. you could not resolve a bug, a visual mismatch, or a physics error), state this clearly and describe what is wrong.
- **If you believe the simulation is correct**, you must prove it with specific evidence from programmatic outputs. Do not simply assert correctness — quote concrete observations from the terminal, rendered frames, or API metadata that confirm the simulation matches the target.
