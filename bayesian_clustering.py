import numpy as np

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
