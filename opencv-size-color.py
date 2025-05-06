import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./download.jpg")
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red, upper_red = np.array([0, 100, 50]), np.array([0, 100, 100])
red_mask = cv2.inRange(hsv_img, lower_red, upper_red)

height, width, _ = img.shape
total_pixels = height * width

detected_img = cv2.bitwise_and(img, img, mask=red_mask)

bgr_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(bgr_img_rgb)
plt.title(f"Banyaknya Pixel Pada Gambar\n {width}x{height} = {total_pixels}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(detected_img_rgb)
plt.title("Deteksi Warna Merah")
plt.axis("off")

plt.show()
