import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_alpha(alpha):
    for row in alpha:
        for val in row:
            if val < 0 or val > 1:
                return False
    return True


# reading input image and trimap
input_image = cv2.imread('Image_Source/Raw_Image/GT04.png');
trimap_image = cv2.imread('Image_Source/Trimap/Trimap1/GT04.png');

# Convert the image and trimap to double precision and normalize
input_image = np.float64(input_image) / 255;
trimap_image = np.float64(trimap_image) / 255;


# size of the image
Row, column, channel = input_image.shape
print("Row, column, channel", Row, column, channel)
print("input image", input_image.shape)
print("trimap image", trimap_image.shape)


# Calculate the foreground, background, and unknown pixels
fg = trimap_image > 0.99
bg = trimap_image < 0.01
unk = ~(fg | bg)

# with open("trimap.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(trimap_image)


# solving using lapalcian method
Laplacian = cv2.Laplacian(input_image, cv2.CV_64F)

# solving for alpha matte
alpha = np.zeros((Row, column))

for i in range(channel):
    squareLap = Laplacian[:, :,i]
    alpha[unk[:, :,i]] += squareLap[unk[:, :,i]]**2

alpha = 1 - np.sqrt(alpha / channel)
alpha[bg[:, :,0]] = 0
alpha[fg[:, :,0]] = 1

# plt.imshow(alpha*255,cmap='gray')
# plt.show()


# with open("alpha.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(alpha)


# write output
cv2.imwrite('Laplacian_output/laplacian_alpha_GT04.png', alpha * 255)






