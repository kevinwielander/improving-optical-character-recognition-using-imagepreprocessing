import requests
import os
import cv2
import numpy as np

input_path = "img/input/test.jpg"
output_path = "img/output"

# Read the input image
img = cv2.imread(input_path)

# Encode the input image into a binary format
img_binary = cv2.imencode('.jpg', img)[1].tobytes()

# Make the request to the API endpoint
response = requests.post("http://localhost:8000/preprocess", data=img_binary)

# Decode the response image binary data
img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)

# Save the response image to the output path
cv2.imwrite(output_path, img)
