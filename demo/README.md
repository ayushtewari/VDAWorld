# VDAWorld demo server

This local web interface uploads an image, starts a generation run, and streams
tool calls and intermediate frames to the browser.

From the repository root:

```bash
uv run python demo/server.py
```

Open <http://127.0.0.1:7860>. Results are written below
`results/demo/<name>/` using the same layout as the command-line pipeline.

The interface includes generation history, rendered output, model tool calls,
and the final simulator source. It executes model-generated Python and has no
authentication. Keep it bound to loopback and use it only with trusted users.
