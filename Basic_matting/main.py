import unittest
from Bayesian import Bayesmat
import laplacian
import cv2
import imageio
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from clustering import clustFunc
from gaussian_filter import gaussian_filter

start_time = time.time()

# Your code goes here

img = imageio.imread("Image_Source\Raw_Image\GT05.png")[:, :, :3]
trimap = imageio.imread("Image_Source\Trimap\Trimap1\GT05.png", as_gray=True)
alpha,F,B = Bayesmat(img, trimap, 25, 8)

cv2.imwrite('Bayesian_output\GT05.png', alpha * 255)

plt.imshow(alpha, cmap='gray')
plt.show()

C = np.zeros(img.shape)
C[:,:,0] = alpha * F[:,:,0] + alpha * B[:,:,0]
C[:,:,1] = alpha * F[:,:,1] + alpha * B[:,:,1]
C[:,:,2] = alpha * F[:,:,2] + alpha * B[:,:,2]

plt.imsave('Basic_matting\composite_output\composite_GT05.png',C)

plt.imshow(C)
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time

print("Time : {:.2f} seconds".format(elapsed_time))

def calculate_weighted_mean_and_covariance(sigma, weights):
    # Calculate weighted mean
    weighted_sum = np.sum(weights)
    weighted_sigma = np.multiply(sigma, weights[:, np.newaxis])
    mean = np.sum(weighted_sigma, axis=0) / weighted_sum
    
    # Calculate weighted covariance
    centered_sigma = sigma - np.broadcast_to(mean, sigma.shape)
    weighted_centered_sigma = np.multiply(centered_sigma, np.sqrt(weights)[:, np.newaxis])
    cov = np.dot(weighted_centered_sigma.T, weighted_centered_sigma) / weighted_sum + 1e-5 * np.eye(sigma.shape[1])
    
    return mean, cov
   

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
        gt_image = imageio.imread("Basic_matting\Image_Source\Ground_Truth\GT05.png")

        # print("gt_image ",gt_image.shape)
        # print("alpha ",alpha.shape)

        height_gt,width_gt,channel_gt = gt_image.shape
        height_alpha,width_alpha = alpha.shape

        self.assertEqual(height_gt,height_alpha, "Height of GT images does not Match with predicted alpha")
        self.assertEqual(width_gt,width_alpha, "Width of GT images does not Match with predicted alpha")
    
    def test_alphavalues(self):
        self.assertEqual(check_alpha(alpha),True,"Alpha values are not Correct")

    #  Testing Guassian filter  
    def test_shape_symmetric(self):
        # Test if the filter is symmetric along both axes
        N = 5
        sigma = 1.0
        g = gaussian_filter(N, sigma)
        for i in range(N):
            for j in range(N):
                self.assertAlmostEqual(g[i,j], g[N-i-1,N-j-1], "The Resultant gaussian matrix is not symmetric")
                
    def test_sum_to_one(self):
        # Test if summation of values should be 1
        N = 5
        sigma = 1.0
        g = gaussian_filter(N, sigma)
        self.assertAlmostEqual(g.sum(), 1.0, "Gaussian sum is not 1.")

    def test_guassian_values(self):
        # Test case 2: N = 5, sigma = 2.0
        g = gaussian_filter(N=5, sigma=2.0)
        GT_g = np.array([[0.023, 0.033, 0.038, 0.033, 0.023],
                            [0.033, 0.049, 0.055, 0.049, 0.033],
                            [0.038, 0.055, 0.062, 0.055, 0.038],
                            [0.033, 0.049, 0.055, 0.049, 0.033],
                            [0.023, 0.033, 0.038, 0.033, 0.023]])

        self.assertAlmostEqual(g.all(), GT_g.all(),"Gaussian values are not Correct")

    def test_clustFunc(self):
        # Sample input data and weights
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        wg = np.array([1, 2, 3])

        # Calculate weighted mean and covariance
        mean, cov = calculate_weighted_mean_and_covariance(X, wg)

        # Test that the output is close to the true values
        GT_mu = np.array([1.88 ,6.64 ,3.58])

        GT_sigma = np.array([[ 0.57  ,0.19 ,0.06],
                             [ 0.19  ,3.87 ,-4.26297578],
                             [ 0.06920415 ,-4.26297578  ,4.83045983] ])

        self.assertAlmostEqual(mean.all(), GT_mu.all(),"clustFunc Mean are not Correct")
        self.assertAlmostEqual(cov.all(), GT_sigma.all(),"clustFunc cov values are not Correct")

    def test_check_Comp_shape(self):
        gt_image = imageio.imread("Basic_matting\Image_Source\Ground_Truth\GT05.png")

        # print("gt_image ",gt_image.shape)
        # print("alpha ",alpha.shape)

        height_gt,width_gt,channel_gt = gt_image.shape
        height_alpha,width_alpha,channel_Comp = C.shape

        self.assertEqual(height_gt,height_alpha, "Height of GT images does not Match with predicted composite")
        self.assertEqual(width_gt,width_alpha, "Width of GT images does not Match with predicted composite")
        self.assertEqual(channel_gt,channel_Comp, "Channel of GT images does not Match with predicted composite")
    

# Run the tests
if __name__ == '__main__':
    unittest.main()