import cv2
import numpy as np


def create_region_of_interest(image: np.ndarray, vertices: np.ndarray):
    """
    Creates a region of interest mask on the given image based on the provided vertices.

    Args:
        image (numpy.ndarray): The input image.
        vertices (numpy.ndarray): List of vertices defining the region of interest.

    Returns:
        numpy.ndarray: The image with the region of interest masked.

    """
    mask: np.ndarray = np.zeros_like(image)
    white_color = 255
    cv2.fillPoly(mask, vertices, white_color)  # type: ignore
    return cv2.bitwise_and(image, mask)
