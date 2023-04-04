import unittest
from Bayesian import Bayesmat
import laplacian
import cv2
import imageio
from tqdm import tqdm
import time
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()

# Your code goes here

img = imageio.imread("Image_Source\Raw_Image\GT12.png")[:, :, :3]
trimap = imageio.imread("Image_Source\Trimap\Trimap1\GT12.png", as_gray=True)
alpha,F,B = Bayesmat(img, trimap, 25, 8)


# plt.imshow(alpha, cmap='gray')
# plt.show()

C = np.zeros(img.shape)
C[:,:,0] = alpha * F[:,:,0] + alpha * B[:,:,0]
C[:,:,1] = alpha * F[:,:,1] + alpha * B[:,:,1]
C[:,:,2] = alpha * F[:,:,2] + alpha * B[:,:,2]

# cv2.imwrite('composite_output\composite_GT12_255.png',c_255)
# cv2.imwrite('composite_output\composite_GT12.png',com)


plt.imsave('Basic_matting\composite_output\composite_GT12.png',C)

# plt.imshow(C)
# plt.show()

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: {:.2f} seconds".format(elapsed_time))

# print("alpha ",alpha.shape)


# Define a function to test
def check_alpha(alpha):
    for row in alpha:
        for val in row:
            if val < 0 or val > 1:
                return False
    return True


# Define a unit test class
class bayesiantesting(unittest.TestCase):
    
    def test_checkshape(self):
        gt_image = imageio.imread("Image_Source\Ground_Truth\GT12.png")

        # print("gt_image ",gt_image.shape)
        # print("alpha ",alpha.shape)

        height_gt,width_gt,channel_gt = gt_image.shape
        height_alpha,width_alpha = alpha.shape

        self.assertEqual(height_gt,height_alpha, "Height of GT images does not Match with predicted alpha")
        self.assertEqual(width_gt,width_alpha, "Width of GT images does not Match with predicted alpha")
    
    def test_alphavalues(self):
        self.assertEqual(check_alpha(alpha),True,"Alpha values are not Correct")

    def test_check_Comp_shape(self):
        gt_image = imageio.imread("Image_Source\Ground_Truth\GT12.png")

        # print("gt_image ",gt_image.shape)
        # print("alpha ",alpha.shape)

        height_gt,width_gt,channel_gt = gt_image.shape
        height_alpha,width_alpha,channel_Comp = C.shape

        self.assertEqual(height_gt,height_alpha, "Height of GT images does not Match with predicted composite")
        self.assertEqual(width_gt,width_alpha, "Width of GT images does not Match with predicted composite")
        self.assertEqual(channel_gt,channel_Comp, "Width of GT images does not Match with predicted composite")
    

    



# Run the tests
if __name__ == '__main__':
    unittest.main()