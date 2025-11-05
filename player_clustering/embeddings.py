import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import gc
from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

V = TypeVar("V")


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence: The input sequence to be batched
        batch_size: Size of each batch (minimum 1)

    Yields:
        Batches of the input sequence
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class EmbeddingExtractor:
    """
    Handles extraction of visual embeddings from player crops using SigLIP model.
    """

    def __init__(self, model_name="google/siglip-base-patch16-224", torso_only=True):
        """
        Initialize the embedding extractor with SigLIP model.

        Args:
            model_name: HuggingFace model name for SigLIP
            torso_only: If True, crop only upper 50% of player bbox (jersey focus)
        """
        self.model = SiglipVisionModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.torso_only = torso_only

    def get_player_crops(self, frame, player_detections):
        """
        Extract player crops from frame using detection bounding boxes.

        Args:
            frame: Input video frame
            player_detections: Player detection results

        Returns:
            List of cropped player images in PIL format
        """
        cropped_images = []
        for boxes in player_detections.xyxy:
            cropped_image = sv.crop_image(frame, boxes)

            # Apply torso-only cropping to focus on jersey colors
            if self.torso_only:
                height = cropped_image.shape[0]
                cropped_image = cropped_image[0:int(height/2), :]  # Upper 50% only

            cropped_images.append(cropped_image)
        cropped_images = [sv.cv2_to_pillow(cropped_image) for cropped_image in cropped_images]
        return cropped_images

    def get_embeddings(self, image_batches):
        """
        Extract SigLIP embeddings from image batches with memory management.

        Args:
            image_batches: Batched images for processing

        Returns:
            Numpy array of embeddings
        """
        data = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(image_batches, desc='extracting_embeddings',
                            mininterval=2.0, ncols=100,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')):
                inputs = self.processor(images=batch, return_tensors="pt").to(device)
                outputs = self.model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

                # Clear GPU cache periodically to prevent memory buildup
                if (batch_idx + 1) % 10 == 0:
                    del inputs, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        data = np.concatenate(data, axis=0)

        # Final cleanup after embedding extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return data
        return data
        return data
        return data