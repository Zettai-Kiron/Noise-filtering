import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_noise(image):
    """Add Gaussian noise to an image."""
    mean = 0
    std = 25
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Load image
image = cv2.imread('BlackWings.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Please ensure 'input_image.jpg' is in the working directory.")

# Add noise
noisy_image = add_noise(image)

# Apply mean and Gaussian filters
kernel_size = 5
mean_filtered = cv2.blur(noisy_image, (kernel_size, kernel_size))
gaussian_filtered = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)

# Plot the results
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title("Noisy")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(mean_filtered, cmap='gray')
plt.title("Mean Filter")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title("Gaussian Filter")
plt.axis('off')

plt.tight_layout()
plt.show()