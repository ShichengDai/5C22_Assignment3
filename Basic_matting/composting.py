import cv2
import matplotlib.pyplot as plt
import numpy as np

foreground = cv2.imread('Image_Source\Raw_Image\GT12.png')
# alpha = cv2.imread('Bayesian_output\GT12.png',0)
background = cv2.imread('Image_Source\plain-green-background.jpg')
gt = cv2.imread('Image_Source\Ground_Truth\GT12.png')
gt = gt/255

# Resize the foreground and alpha 
# foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))
# alpha = cv2.resize(alpha, (background.shape[1], background.shape[0]))

# # creating a mask
# mask = cv2.merge([alpha, alpha, alpha])

# # Inverting the mask
# inverse_mask = cv2.bitwise_not(mask)

# # Extracting the foreground from background
# fg = cv2.bitwise_and(foreground, mask)
# bg = cv2.bitwise_and(background, inverse_mask)

# # Adding the foreground and background
# comp_img = cv2.add(fg, bg)
background = cv2.resize(background, (foreground.shape[1],foreground.shape[0]))

F = np.zeros(background.shape)
B = np.zeros(background.shape)

F[:,:,0] = foreground[:,:,2] * gt[:,:,0]
F[:,:,1] = foreground[:,:,1] * gt[:,:,1]
F[:,:,2] = foreground[:,:,0] * gt[:,:,2]

B[:,:,0] = background[:,:,0] * (1 - gt[:,:,0])
B[:,:,1] = background[:,:,1] * (1 - gt[:,:,1])
B[:,:,2] = background[:,:,2] * (1 - gt[:,:,2])

comp_img = F + B

# comp_img = F.astype(np.uint8) + B.astype(np.uint8)

plt.imsave('Image_Source\Const_bg_Raw\GT12.png', comp_img/255)
