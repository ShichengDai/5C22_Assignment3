function [F,B,alpha] = Bayesmat(im, trimap, N, sigma)
% im and trimap should be double
% im = m * n * 3  trimap = m * n
% N is the length of the edges of the neibourhood
% N should be an even number
% Sigma is a parameter of gaussian filter

imsize = size(trimap);

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

%make the number of unkown pixels
NumUnk = sum(unkreg(:));

%make a caounter to do the loop
n = 1;

while n < NumUnk

    % find the locations for each unkown pixel
    [Y, X] = find(unkreg);

    for i = 1 : length(Y)

        %make the parameter c here
        c = reshape(im(Y(i),X(i),:),[3,1]);

        %make a window to compute for F, B and Alpha
        a = makewindow(Y(i), X(i), N, Alpha);
        fgraph = makewindow(Y(i), X(i), N, F);
        bgraph = makewindow(Y(i), X(i), N, B);

        %calculate for weights
        fw = (a .^ 2) .* g;
        fw = fw(fw > 0);
        bw = ((1 - a) .^ 2) .* g;
        bw = bw(bw > 0);
        if length(fw)<Nthres || length(bw)<Nthres
            continue;
        end
        %Do clustering and solving here

    end

end


