import cv2
import numpy as np

# read input image and trimap
img = cv2.imread('donkey_input.png');
trimap = cv2.imread('donkey_trimap.png');

# Convert the image and trimap to double precision and normalize
img = np.float64(img);
trimap = np.float64(trimap);
# trimap = trimap / 255

# size of the image
m, n, c = img.shape

# Calculate the foreground, background, and unknown pixels
fg = trimap > 0.99
bg = trimap < 0.01
unk = ~(fg | bg)

# solving using lapalcian method
Laplacian = cv2.Laplacian(img, cv2.CV_64F)

# solving for alpha matte
alpha = np.zeros((m, n))
for i in range(c):
    alpha[unk[:, :, 0]] += np.square(Laplacian[unk[:, :, 0], i])
alpha = 1 - np.sqrt(alpha / c)
alpha[bg[:, :, 0]] = 0
alpha[fg[:, :, 0]] = 1

# write output
cv2.imwrite('output_alpha.png', alpha * 255)