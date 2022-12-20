import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the gradient in the x and y directions
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the surface normals
    z = np.ones_like(gray)
    nx = -dx / z
    ny = -dy / z
    nz = np.sqrt(1 - nx**2 - ny**2)

    # Display the surface normals
    cv2.imshow('Surface Normals', np.stack((nx, ny, nz), axis=2))

    # Check for user input
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
