import logging
import os
import sys
import cv2
import numpy as np
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.edge_detection import create_canny_edge_detection_img
from utils.hough_transform import hough_transform, get_hough_detected_lines_image
from utils.region_of_interest import create_region_of_interest_mask


script_directory = os.path.dirname(__file__)
project_directory = os.path.abspath(os.path.join(script_directory, "..", ".."))
image_directory = os.path.join(project_directory, "images")

LOGGER = logging.getLogger(__name__)


def main():
    st.set_page_config(page_title="Lane Detection App", page_icon=":car:")

    st.title("🛣️ Lane Detection")

    # Set up the image selection
    image_files = [f for f in os.listdir(image_directory)]
    selected_image = str(st.selectbox("Select an image", image_files, index=0))
    image_path = os.path.join(image_directory, selected_image)
    image = cv2.imread(image_path)

    # Set up the sidebar for parameter sliders
    st.sidebar.title("⚙️ Adjust Parameters")

    st.sidebar.subheader("Canny Edge Detection Parameters")
    lower_threshold = st.sidebar.slider("🔽 Lower Threshold", 0, 255, 100)
    upper_threshold = st.sidebar.slider("🔼 Upper Threshold ", 0, 255, 200)

    st.sidebar.subheader("Hough Transform Parameters")
    rho = st.sidebar.slider("Rho", 1, 10, 1)
    theta = st.sidebar.slider("Theta", 0.0, np.pi / 2, np.pi / 180)
    threshold = st.sidebar.slider("Threshold", 1, 500, 100)
    min_line_length = st.sidebar.slider("Min Line Length", 1, 500, 100)
    max_line_gap = st.sidebar.slider("Max Line Gap", 1, 100, 10)

    # Display the results
    st.subheader("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Perform Canny edge detection on the ROI image
    edges = create_canny_edge_detection_img(image, lower_threshold, upper_threshold)

    st.subheader("Edges")
    st.image(edges, use_column_width=True)

    # Perform Hough transform on the edges
    lines_and_edges_image = hough_transform(
        edges, rho, theta, threshold, min_line_length, max_line_gap
    )

    lines_image = get_hough_detected_lines_image(
        edges, rho, theta, threshold, min_line_length, max_line_gap
    )

    st.subheader("Detected Lines")
    st.image(cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Define the vertices of the ROI polygon
    height, width = image.shape[:2]
    roi_vertices = np.array(
        [[(0, height), (width // 2, 0), (width, height)]], dtype=np.int32
    )
    # Create the ROI mask
    roi_mask = create_region_of_interest_mask(image, roi_vertices)

    st.subheader("ROI Mask")
    st.image(roi_mask, use_column_width=True)

    # Apply the ROI mask to the image
    roi_image = cv2.bitwise_and(image, image, mask=roi_mask)

    st.subheader("ROI Image")
    st.image(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Combine the line image with the ROI image
    combined_image = cv2.addWeighted(roi_image, 0.8, lines_and_edges_image, 1, 0)

    st.subheader("Combined Image")
    st.image(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Edges and Lines with mask applied
    refined_lines_and_edges_image = lines_and_edges_image.copy()
    refined_detected_lines_and_edges = cv2.bitwise_and(
        refined_lines_and_edges_image, refined_lines_and_edges_image, mask=roi_mask
    )
    st.subheader("Edgeds and Lines with mask applied")
    st.image(
        cv2.cvtColor(refined_detected_lines_and_edges, cv2.COLOR_BGR2RGB),
        use_column_width=True,
    )

    # Lines with mask applied
    refined_lines_image = lines_image.copy()
    refined_detected_line = cv2.bitwise_and(
        refined_lines_image, refined_lines_image, mask=roi_mask
    )
    st.subheader("Lines with mask applied")
    st.image(
        cv2.cvtColor(refined_detected_line, cv2.COLOR_BGR2RGB), use_column_width=True
    )

    # Print original image with refined detected lines
    refined_detected_line_color = cv2.cvtColor(
        refined_detected_line, cv2.COLOR_GRAY2BGR
    )
    # Create a mask for the non-zero pixels (lines)
    line_mask = cv2.inRange(refined_detected_line, 1, 255)  # type: ignore
    # Create a bright red color image
    bright_red = np.zeros_like(refined_detected_line_color)
    bright_red[line_mask != 0] = [0, 0, 255]  # Bright red color (B, G, R)

    # Combine the original image with the colored refined_detected_line
    combined_refined_image = cv2.addWeighted(image, 1, bright_red, 2, 1)
    st.subheader("Masked Original Image with Detected Lines")

    st.image(
        cv2.cvtColor(combined_refined_image, cv2.COLOR_BGR2RGB), use_column_width=True
    )


if __name__ == "__main__":
    LOGGER.setLevel(logging.DEBUG)
    main()
