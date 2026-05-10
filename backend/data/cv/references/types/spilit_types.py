import cv2
import os

# Load image
image = cv2.imread("types.png")  # <-- rename your file to types.png

# Get dimensions
height, width, _ = image.shape

# Grid size (based on your image)
rows = 3
cols = 6

# Calculate size of each icon
icon_h = height // rows
icon_w = width // cols

# Output folder
output_dir = "type_icons"
os.makedirs(output_dir, exist_ok=True)

# Type names in order (left to right, top to bottom)
types = [
    "normal", "fire", "water", "grass", "electric", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy"
]

index = 0

for r in range(rows):
    for c in range(cols):
        y1 = r * icon_h
        y2 = (r + 1) * icon_h
        x1 = c * icon_w
        x2 = (c + 1) * icon_w

        icon = image[y1:y2, x1:x2]

        filename = f"{types[index]}.png"
        cv2.imwrite(os.path.join(output_dir, filename), icon)

        print(f"Saved {filename}")
        index += 1