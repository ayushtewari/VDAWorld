"""SAM 3 image segmentation adapter."""

import numpy as np
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def segment(image: Image.Image, object_prompt: str) -> np.ndarray:
    """Return masks for all instances matching `object_prompt`."""

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(
        state=inference_state,
        prompt=object_prompt,
    )
    return output["masks"].squeeze(1).detach().cpu().numpy()
