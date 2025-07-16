import cv2
import numpy as np
import cv2.aruco as aruco

# Parameters
squares_x = 5   # number of squares along X
squares_y = 5   # number of squares along Y
square_length = 60  # in mm (or any consistent unit)
marker_length = 45  # marker side length
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


# Create ChArUco board
board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

# Draw the board to an image
img = board.generateImage((squares_x * square_length, squares_y * square_length))

# Save as image
cv2.imwrite(f"charuco_board.png", img)

# Optional: Convert to PDF
import matplotlib.pyplot as plt

plt.figure(figsize=(8.5, 11))  # A4 size page
plt.axis('off')
plt.imshow(img, cmap='gray')

plt.savefig(f"charuco_board_{squares_x}x{squares_y}.pdf", bbox_inches='tight', pad_inches=0, dpi=2400)
plt.savefig(f"charuco_board_{squares_x}x{squares_y}.svg", bbox_inches='tight', pad_inches=0, dpi=2400)
