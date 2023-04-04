import cv2

foreground = cv2.imread('Image_Source\Raw_Image\GT01.png')
alpha = cv2.imread('Bayesian_output\GT01.png',0)
background = cv2.imread('background.png')

# Resize the foreground and alpha 
foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))
alpha = cv2.resize(alpha, (background.shape[1], background.shape[0]))

# creating a mask
mask = cv2.merge([alpha, alpha, alpha])

# Inverting the mask
inverse_mask = cv2.bitwise_not(mask)

# Extracting the foreground from background
fg = cv2.bitwise_and(foreground, mask)
bg = cv2.bitwise_and(background, inverse_mask)

# Adding the foreground and background
comp_img = cv2.add(fg, bg)
cv2.imwrite('image.png', comp_img)