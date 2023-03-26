import numpy as np
import cv2 


# Load the images
Gt_img = cv2.imread('gt.png');
predicted_img = cv2.imread('result.png');

Gt_img = Gt_img[:,:,1]/255;
predicted_img = predicted_img[:,:,1]/255;

print(predicted_img.shape);
print(Gt_img.shape);

def check_alpha(alpha):
    for row in alpha:
        for val in row:
            if val < 0 or val > 1:
                return False
    return True


def cal_mse():

    # Calculate mse
    diff = (Gt_img - predicted_img) ** 2
    mse = np.sum(diff) / (Gt_img.shape[0] * Gt_img.shape[1])

    print("MSE:", mse);
    return mse


def psnr(Gt_img, predicted_img = cv2.imread('result.png')):
    # calculate MSE 
    diff = (Gt_img - predicted_img) ** 2
    mse = np.sum(diff) / (Gt_img.shape[0] * Gt_img.shape[1])
    
    # calculate PSNR 
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    
    return psnr

# compare size of GT and Predicted alpha
if Gt_img.shape == predicted_img.shape:
    print("Correct Sizes.");
    if check_alpha(predicted_img):

        # calculate MSE
        mse = cal_mse();
        print("MSE :", mse)

        # calculate the PSNR
        psnr_value = psnr(Gt_img, predicted_img)

        # PSNR value
        print("PSNR :", psnr_value, "dB")
    else:
        print("Alpha values are not correct!!!");
else:
    print("Error Diffrent sizes.")








