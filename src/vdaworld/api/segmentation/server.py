"""FastAPI service for SAM 3 image segmentation."""

import gc
import io
import logging

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from vdaworld.api.segmentation.segment import segment

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(name)s: %(message)s",
)

app = FastAPI(title="VDAWorld SAM 3 Segmentation Service")
logger = logging.getLogger(__name__)


@app.post("/segment")
async def run_segmentation_endpoint(
    image: UploadFile = File(...),
    object_prompt: str = Form(...),
) -> Response:
    """Accept an image and text prompt and return NumPy mask data."""

    try:
        gc.collect()
        pil_img = Image.open(io.BytesIO(await image.read())).convert("RGB")
        logger.info("Segmenting %r on image size %s", object_prompt, pil_img.size)

        masks = segment(pil_img, object_prompt)
        out_buffer = io.BytesIO()
        np.save(out_buffer, masks)
        gc.collect()
        return Response(
            content=out_buffer.getvalue(),
            media_type="application/octet-stream",
        )
    except Exception as exc:
        logger.exception("Segmentation failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
