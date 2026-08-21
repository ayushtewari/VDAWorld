# External model services

VDAWorld calls GPU-heavy vision models through local HTTP services. This keeps
their dependencies out of the core package and allows each model to use an
independent CUDA, PyTorch, and Python stack.

## Upstream projects

- [SAM 3](https://github.com/facebookresearch/sam3) provides text-prompted image
  segmentation. Install it in the `sam3` Conda environment using its upstream
  instructions.
- [MoGe](https://github.com/microsoft/MoGe) is the default point-map backend.
  Clone it into a separate checkout, switch to the commit recorded in your
  deployment manifest, and install that checkout inside the `moge` environment.
- [VGGT](https://github.com/facebookresearch/vggt) is an alternative point-map
  backend. Enable `depth_vggt` and disable `depth_moge` in `api_config.yaml`,
  then set `VDAWORLD_DEPTH_BACKEND=vggt`. Note that by default, this module is not used and not required.
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) provides
  mesh generation. Set `SAM3D_PATH` to its checkout after following the
  upstream checkpoint instructions.

Review every upstream license and model license for the intended use. These
projects and their weights are not part of the VDAWorld package.

Each service environment also needs FastAPI, Uvicorn, python-multipart, Pillow,
NumPy, and the VDAWorld source importable. The launcher injects this
repository's `src/` directory into `PYTHONPATH` for child services.

## Launcher configuration

`api_config.yaml` records the service layout used for the published work. Copy
it to `api_config.local.yaml` and adjust the environment names and GPU choices
for your machine. vLLM is disabled by default.

The `vllm` service runs `vllm serve` from the Conda environment named by
`conda_env`. A `vllm_project_path` is not required: install a compatible vLLM
release in that environment, or install a source checkout into it in editable
mode. `tensor_parallel_size` is optional and defaults to one; set it to the
number of visible GPUs when the model is sharded with tensor parallelism. The
checked-in Qwen3-VL example exposes four GPUs and therefore uses a value of
four.

```bash
uv run vdaworld-start-api --config api_config.local.yaml
```

Services use loopback addresses. The core API accepts these environment
variable overrides:

| Variable                    | Default                                    | Service        |
| --------------------------- | ------------------------------------------ | -------------- |
| `VDAWORLD_SEGMENTATION_URL` | `http://127.0.0.1:5001/segment`            | SAM 3          |
| `VDAWORLD_MOGE_URL`         | `http://127.0.0.1:5004/estimate_3d_points` | MoGe           |
| `VDAWORLD_VGGT_URL`         | `http://127.0.0.1:5003/estimate_3d_points` | VGGT           |
| `VDAWORLD_MESH_URL`         | `http://127.0.0.1:5005/estimate_3d_mesh`   | SAM 3D Objects |

Do not expose these development services directly to an untrusted network.
Put authentication, request limits, and input-size limits in front of them for
any shared deployment.
