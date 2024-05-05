import cv2
import numpy as np


def create_canny_edge_detection_img(
    image: np.ndarray,
    lower_threshold: int = 100,
    upper_threshold: int = 200,
    gaussian_kernel_size=(5, 5),
    sigma_standard_deviation=0,
) -> np.ndarray:
    """
    Apply Canny edge detection to the input image.

    Parameters:
        image (np.ndarray): The input image.
        lower_threshold (int): The lower threshold value for the Canny edge detection algorithm. Default is 100.
        upper_threshold (int): The upper threshold value for the Canny edge detection algorithm. Default is 200.
        gaussian_kernel_size (tuple): The size of the Gaussian kernel used for blurring the image. Default is (5, 5).
        sigma_standard_deviation (int): The standard deviation of the Gaussian kernel used for blurring the image. Default is 0.

    Returns:
        np.ndarray: The edge-detected image.

    """
    # Check if the input image is grayscale
    if len(image.shape) != 2:
        # Convert the image to grayscale if it's not already
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(image, gaussian_kernel_size, sigma_standard_deviation)
    edges: np.ndarray = cv2.Canny(blurred, lower_threshold, upper_threshold)
    return edges
