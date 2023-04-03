import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import scipy.io as sio

plt.close('all')


#%%
def FandB(f_var, b_var, f_mean, b_mean, un_c, alpha):
    SigmaC_Sq = 64
    f_inv = np.linalg.inv(f_var)
    b_inv = np.linalg.inv(b_var)
    A_fl = f_inv + (np.eye(3) * alpha * alpha) / (SigmaC_Sq)
    A_fr = (np.eye(3) * alpha * (1 - alpha)) / (SigmaC_Sq)
    A_sl = (np.eye(3) * alpha * (1 - alpha)) / (SigmaC_Sq)
    A_sr = b_inv + (np.eye(3) * (1 - alpha) * (1 - alpha)) / (SigmaC_Sq)
    A = np.vstack((np.hstack((A_fl, A_fr)), np.hstack((A_sl, A_sr))))
    b_l = (f_inv @ f_mean[:, None]) + (un_c * alpha) / (SigmaC_Sq)
    b_r = (b_inv @ b_mean[:, None]) + (un_c * (1 - alpha)) / (SigmaC_Sq)
    b = np.concatenate((b_l, b_r))

    x = np.linalg.solve(A, b)
    F = x[:3]
    B = x[3:]

    return F, B


#%%
imgo = Image.open('gandalf-input.png')
trimap = Image.open('gandalf-trimap.png')
trimap = trimap.convert('L')



plt.figure()
plt.imshow(imgo)
plt.axis('off')
plt.figure()
plt.imshow(trimap, cmap='gray')



# Convert imgo and trimap to double precision
imgo = np.array(imgo).astype(np.float64) / 255.0
trimap = np.array(trimap).astype(np.float64) / 255.0

M, N, _ = imgo.shape


# Initialize output arrays
fg = np.zeros((M, N, 3))
bg = np.zeros((M, N, 3))
ug = np.zeros((M, N, 3))

# Loop over each pixel and assign values based on trimap value
for i in range(M):
    for j in range(N):
        if trimap[i, j] == 1:
            fg[i, j, :] = imgo[i, j, :]
        elif trimap[i, j] == 0:
            bg[i, j, :] = imgo[i, j, :]
        else:
            ug[i, j, :] = imgo[i, j, :]
plt.figure()
plt.imshow(np.hstack([fg, bg, ug]))
plt.axis('off')
# Compute statistics from known foreground & background
# Finding mean of known foreground & background
f_mean = np.zeros(3)
b_mean = np.zeros(3)

for i in range(3):
    f = fg[:, :, i]
    b = bg[:, :, i]
    f_mean[i] = np.mean(f[np.where(f)])
    b_mean[i] = np.mean(b[np.where(b)])

# Finding variance of known foreground
f_div = np.zeros_like(fg)
f_div[:, :, 0] = fg[:, :, 0] - f_mean[0]
f_div[:, :, 1] = fg[:, :, 1] - f_mean[1]
f_div[:, :, 2] = fg[:, :, 2] - f_mean[2]

sumF = np.zeros((3, 3))
count = 0

for c in range(f_div.shape[1]):
    for r in range(f_div.shape[0]):
        pixF = f_div[r, c, :]
        pixF = pixF.reshape((3, 1))
        if np.any(fg[r, c, :]):
            sumF += pixF @ pixF.T
            count += 1

f_var = sumF / count

# finding variance of known background
b_div = np.zeros_like(bg)
b_div[:, :, 0] = bg[:, :, 0] - b_mean[0]
b_div[:, :, 1] = bg[:, :, 1] - b_mean[1]
b_div[:, :, 2] = bg[:, :, 2] - b_mean[2]

sumB = np.zeros((3, 3))
count = 0

for c in range(b_div.shape[1]):
    for r in range(b_div.shape[0]):
        pixB = b_div[r, c, :]
        pixB = pixB.reshape((3, 1))
        if np.any(bg[r, c, :]):
            sumB += pixB @ pixB.T
            count += 1

b_var = sumB / count

# Solve for unknown pixel values and alpha
# initiating alpha as avg of neighbors
alpha_un = trimap
alphaAvg = convolve(alpha_un, np.ones((3, 3)) / 9)
M, N, _ = ug.shape

for i in range(M):
    for j in range(N):
        if not np.any(ug[i, j, :]):
            continue

        un_c = ug[i, j, :].reshape((3, 1))
        alpha = alphaAvg[i, j]

        # iteratively solve for color of pixel
        for k in range(10):
            alpha_prev = alpha
            F, B = FandB(f_var, b_var, f_mean, b_mean, un_c, alpha)
            alpha = np.dot((un_c - B).flatten(), (F - B).flatten()) / np.linalg.norm(F - B)**2

            if abs(alpha - alpha_prev) <= 0.0001:
                break
        print(alpha)
        alpha_un[i, j] = alpha
        fg[i, j, :] = F.flatten()
        bg[i, j, :] = B.flatten()
#sio.savemat('alpha_un.mat', {'alpha_un': alpha_un})

print(alpha_un.shape)
plt.figure()
plt.imshow(alpha_un)
plt.axis('off')
plt.title('Alpha Matte')


# fuse
img3 = Image.open('gandalf-background.png')
img3 = np.array(img3).astype(np.float64) / 255.0

img_final = np.copy(img3)

img_final[:, :, 0] = fg[:, :, 0] * alpha_un[:, :] + img3[:, :, 0] * (1 - alpha_un[:, :])
img_final[:, :, 1] = fg[:, :, 1] * alpha_un[:, :] + img3[:, :, 1] * (1 - alpha_un[:, :])
img_final[:, :, 2] = fg[:, :, 2] * alpha_un[:, :] + img3[:, :, 2] * (1 - alpha_un[:, :])
img_final = np.clip(img_final, 0, 1)

plt.figure()
plt.imshow(img_final)
plt.axis('off')
plt.title('Composed Image')
plt.show()