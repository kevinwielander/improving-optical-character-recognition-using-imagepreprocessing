import cv2
from PIL import Image


def check_and_scale_dpi(image_path):
    # Open the image using PIL and get its DPI
    with Image.open(image_path) as img:
        dpi = img.info.get('dpi')

    # If the DPI is not available, assume it's 72 DPI
    if dpi is None:
        dpi = (72, 72)

    # Calculate the current DPI of the image
    current_dpi = max(dpi)

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # If the current DPI is less than 300, scale the image to 300 DPI
    if current_dpi < 300:
        # Calculate the scaling factor to convert the image to 300 DPI
        scaling_factor = 300.0 / current_dpi

        # Resize the image using OpenCV
        resized_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)

        # Save the resized image with 300 DPI
        cv2.imwrite(image_path, resized_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
