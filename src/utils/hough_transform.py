import cv2
import numpy as np


def hough_transform(
    edges: np.ndarray,  # output of the Canny edge detector
    rho: float = 1,  # distance resolution in pixels of the Hough grid
    theta: float = np.pi / 180,  # angular resolution in radians of the Hough grid,
    threshold: int = 100,  # minimum number of votes (intersections in Hough grid cell)
    min_line_length: int = 100,  # minimum number of pixels making up a line
    max_line_gap: int = 10,  # maximum gap in pixels between connectable line segments
) -> np.ndarray:

    # %%
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    # Create a copy of the original image to draw lines on
    line_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Draw lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return line_image
