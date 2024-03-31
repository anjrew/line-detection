import logging
import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def create_region_of_interest_mask(image: np.ndarray, vertices: np.ndarray):
    """
    Creates a region of interest mask on the given image based on the provided vertices.

    Args:
        image (numpy.ndarray): The input image.
        vertices (numpy.ndarray): List of vertices defining the region of interest.

    Returns:
        numpy.ndarray: The image with the region of interest masked.

    """
    LOGGER.debug("Creating region of interest mask")
    roi_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [vertices], 255)  # type: ignore
    return roi_mask
