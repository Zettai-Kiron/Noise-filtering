import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_noise(image, noise_type="gaussian", mean=0, std=25):
    """Add noise to an image."""
    if noise_type == "gaussian":
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    elif noise_type == "salt_pepper":
        noisy_image = image.copy()
        salt_pepper_ratio = 0.02
        num_salt = int(salt_pepper_ratio * image.size * 0.5)
        coords = [
            np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]
        ]
        noisy_image[coords[0], coords[1]] = 255

        num_pepper = int(salt_pepper_ratio * image.size * 0.5)
        coords = [
            np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]
        ]
        noisy_image[coords[0], coords[1]] = 0
        return noisy_image

# Load image
image = cv2.imread('flag.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Please ensure 'flag.jpg' is in the working directory.")

# Add Gaussian noise
noisy_image = add_noise(image, noise_type="gaussian")

# Apply mean and Gaussian filters with varying kernel sizes
kernel_sizes = [3, 5, 7]
results = {"Original": image, "Noisy": noisy_image}

for size in kernel_sizes:
    mean_filtered = cv2.blur(noisy_image, (size, size))
    gaussian_filtered = cv2.GaussianBlur(noisy_image, (size, size), 0)

    results[f"Mean Filter (Kernel {size})"] = mean_filtered
    results[f"Gaussian Filter (Kernel {size})"] = gaussian_filtered

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
