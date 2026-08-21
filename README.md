# VDAWorld: World Modelling via VLM-Directed Abstraction and Simulation

_This is the official repository for [VDAWorld: World Modelling via VLM-Directed Abstraction and Simulation](https://felixomahony.github.io/vdaworld) (SIGGRAPH Asia 2026)._

<video controls>
  <source src="docs/source_grid.mp4" type="video/mp4">
</video>

Generative video models, a leading approach to world modeling, face fundamental limitations. They often violate physical and logical rules, lack interactivity, and operate as opaque black boxes ill-suited for building structured, queryable worlds. To overcome these challenges, we propose a new paradigm focused on distilling an image caption pair into a tractable, abstract representation optimized for simulation. We introduce VDAWorld, a framework where a Vision-Language Model (VLM) acts as an intelligent agent to orchestrate this process. The VLM autonomously constructs a grounded (2D or 3D) scene representation by selecting from a suite of vision tools, and co-dependently chooses a compatible physics simulator (e.g., rigid body, fluid) to act upon it. Furthermore, VDAWorld can infer latent dynamics from the static scene to predict plausible future states. Our experiments show that this combination of intelligent abstraction and adaptive simulation results in a versatile world model capable of producing higher-quality simulations across a wider range of dynamic scenarios than prior approaches.

## How it works

1. A YAML configuration selects an input, prompt set, model alias, and output
   directory.
2. The generation agent writes `simulator_gen.py` and iteratively checks it in
   a subprocess sandbox.
3. The agent can query locally hosted segmentation, depth, and mesh services.
4. The pipeline exports the final source, rendered frames, video, tool logs,
   and model interaction records.

## Requirements

- Python 3.13
- `uv` (recommended) or another PEP 517-compatible installer
- FFmpeg for video output
- A supported VLM API key or a local OpenAI-compatible VLM endpoint
- Optional Conda environments for the external vision services

## Installation

```bash
uv sync --group dev
```

For a runtime-only editable installation:

```bash
python -m pip install -e .
```

Set the credential required by the selected model backend:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

Gemini model aliases included in `models/` use the Gemini Developer API. To
use Vertex AI instead, create a local model alias with `backend: gemini` and
set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.

## External vision services

SAM 3, MoGe or VGGT, and SAM 3D Objects have large and sometimes conflicting
runtime dependencies. Install them in separate Conda environments, then adjust
the generic environment names and GPU selections in `api_config.yaml`.

```bash
uv run vdaworld-start-api
```

The services bind to `127.0.0.1` by default. See
[`docs/external-services.md`](docs/external-services.md) for installation,
version pinning, and endpoint configuration. A task can run without these
tools by setting `task.no_api: true`.

## Run the pipeline

The included demo configuration uses an image committed under `example_data/`:

```bash
uv run vdaworld --config configs/demo.yaml
```

For a local vLLM run, enable the `vllm` service in a local copy of
`api_config.yaml`, set its `conda_env` and GPU options for your machine, start
the services, and use the matching example:

```bash
uv run vdaworld --config configs/demo_vllm.yaml
```

The Conway's Game of Life example uses the included glider image:

```bash
uv run vdaworld --config configs/demo_conway.yaml
```

Other hosted providers do not require duplicate task configurations. Copy
`configs/demo.yaml` and change `vlm.model` to another alias from `models/`,
such as `openai_gpt55` or `claude_opus48`.

Configuration paths are resolved relative to the repository root. Exactly one
of `task.image_path` and `task.video_path` is required. Model aliases live in
`models/<alias>.yaml`; prices in those files are estimates used only for the
run summary and should be reviewed before relying on them.

The configured output directory is created with a `.vdaworld-output` marker.
On a later run, VDAWorld only clears a non-empty output directory when that
marker is present. This guard helps prevent an accidental recursive deletion
after a configuration typo.

Use `--load-existing` to skip generation and rerun the simulator already stored
in the configured output directory:

```bash
uv run vdaworld --config configs/demo.yaml --load-existing
```

## Local interfaces

The generation demo accepts an input image and streams progress:

```bash
uv run python demo/server.py
```

The results explorer reads generated files under `results/`:

```bash
uv run vdaworld-results
```

Both interfaces bind to loopback unless explicitly configured otherwise.

## Repository layout

```text
api_config.yaml       Local model-service launcher configuration
configs/              Reproducible pipeline configurations
demo/                 Local generation web interface
docs/                 Deployment and dependency notes
example_data/         Small demonstration inputs
models/               VLM backend aliases and cost estimates
prompts/              Agent instructions and simulator specifications
scripts/              Publication and metadata checks
src/vdaworld/         Python package
tests/                 Unit tests
```

Generated results, credentials, machine-local configuration, virtual
environments, and assistant/editor state are excluded by `.gitignore`.

## Development checks

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
uv run python scripts/sanitize_image_metadata.py --check example_data
uv build
```

## Citation

```
@inproceedings{omahony2026vdaworld,
  title     = {VDAWorld: World Modelling via VLM-Directed Abstraction and Simulation},
  author    = {O'Mahony, Felix and Cipolla, Roberto and Tewari, Ayush},
  booktitle = {Proceedings of SIGGRAPH Asia},
  year      = {2026}
}
```
