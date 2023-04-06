import cv2
import matplotlib.pyplot as plt
import numpy as np

foreground = cv2.imread('Image_Source\Raw_Image\GT04.png')
alpha = cv2.imread('Bayes_Output\const_GT04.png',0)
background = cv2.imread('Image_Source/bg.jpg')
alpha = alpha/255
# gt = cv2.imread('Image_Source\Ground_Truth\GT12.png')
# gt = gt/255

# Resize the foreground and alpha 
foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))
alpha = cv2.resize(alpha, (background.shape[1], background.shape[0]))



# # creating a mask
# mask = cv2.merge([alpha, alpha, alpha])

# # # Inverting the mask
# inverse_mask = cv2.bitwise_not(mask)

# # # Extracting the foreground from background
# fg = cv2.bitwise_and(foreground, mask)
# bg = cv2.bitwise_and(background, inverse_mask)

# # # Adding the foreground and background
# comp_img = cv2.add(fg, bg)
background = cv2.resize(background, (foreground.shape[1],foreground.shape[0]))

F = np.zeros(background.shape)
B = np.zeros(background.shape)

F[:,:,0] = foreground[:,:,2] * alpha
F[:,:,1] = foreground[:,:,1] * alpha
F[:,:,2] = foreground[:,:,0] * alpha

B[:,:,0] = background[:,:,0] * (1 - alpha)
B[:,:,1] = background[:,:,1] * (1 - alpha)
B[:,:,2] = background[:,:,2] * (1 - alpha)

comp_img = F + B

# comp_img = F.astype(np.uint8) + B.astype(np.uint8)

plt.imsave('Basic_matting\composite_output\comp_GT04_const.png', comp_img/255)
