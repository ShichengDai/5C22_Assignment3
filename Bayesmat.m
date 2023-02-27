function [F,B,alpha] = Bayesmat(im, trimap, N, sigma)
% im and trimap should be double
% im = m * n * 3  trimap = m * n
% N is the size of the neibourhood
% Sigma is a parameter of gaussian filter


%make masks for fg bg and unkreg
bgmask2D = trimap == 0;
bgmask3D(:, :, 1 : 3) = bgmask2D;
fgmask2D = trimap == 1;
fgmask3D(:, :, 1 : 3) = fgmask2D;
unkreg = trimap ~= 1 && trimap ~= 0;

%initialize F B and Alpha
F = im;
B = im;
Alpha = zeros(size(trimap));
F(fgmask3D) = 0;
B(bgmask3D) = 0;
Alpha(fgmask2D) = 1;
Alpha(bgmask2D) = 0;
Alpha(unkreg) = NaN;

%make gasussian parameter p
g = fspecial('gaussian', N, sigma);

%mormalize the parameter to make sure p will not change the image luminance
g = g / max(g(:));

%set a threshold for the minimum valid pixels in the neibourhood
Nthres = 10;


