im = imread('GT04.png');
trimap = imread('trimap-GT04.png');
alpha0 = imread('alpha0-GT04.png');
conf0 = imread('conf-GT04.png');
gt = imread('GT04-GT.png');
gt = gt(:, :, 1);

im = im2double(im);
trimap = im2double(trimap);
alpha0 = im2double(alpha0);
conf0 = im2double(conf0);
gt = im2double(gt);

% set the size of window
N = 25;

%set the value of gaussian parameter
sigma = 8;

[F, B, alpha1] = Bayesmat(im, trimap, N, sigma);

figure(1);
imshow(alpha1);
title('Alpha Bayesian Matting');

% do Laplacian matting
walpha0 = exp(log(conf0)*3);
walpha0(trimap < .1) = 2;
walpha0(trimap > .9) = 2;

[alpha2, beta] = laplacian(...
    im, 'alpha0', alpha0, 'walpha0', walpha0, 'sigma_r', 1, 'T', .001);

figure(1);
imshow(alpha2);
title('Alpha Laplacian Matting');

mse = mean(sum((gt - alpha1) .^ 2));
mse2 = mean(sum((gt - alpha2) .^ 2));

