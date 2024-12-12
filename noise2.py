import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('flag.jpg', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noisy_image = add_noise(image, noise_type="gaussian")

# Apply mean and Gaussian filters with varying kernel sizes
kernel_sizes = [3, 5, 7]
results = {"Original": image, "Noisy": noisy_image}

# Plot the results
plt.figure(figsize=(15, 10))
num_results = len(results)
for i, (title, img) in enumerate(results.items()):
    plt.subplot(2, (num_results + 1) // 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()