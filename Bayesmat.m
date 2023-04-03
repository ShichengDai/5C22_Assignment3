function [F,B,alpha] = Bayesmat(im, trimap, N, sigma)
% % im and trimap should be double
% % im = m * n * 3  trimap = m * n
% % N is the length of the edges of the neibourhood
% % N should be an even number
% % Sigma is a parameter of gaussian filter

%make masks for fg bg and unkreg
bgmask = trimap == 0; 
fgmask = trimap == 1; 
unkmask = ~bgmask & ~fgmask;

%initialize F B and Alpha
F = im; 
F(repmat(~fgmask, [1, 1, 3])) = 0;
B = im; 
B(repmat(~bgmask, [1, 1, 3])) = 0;
alpha = zeros(size(trimap));
alpha(fgmask) = 1;
alpha(unkmask) = NaN;

nUnknown = sum(unkmask(:));

%make gasussian parameter g
g = fspecial('gaussian', N, sigma); 
%mormalize the parameter to make sure p will not change the image luminance
g = g / max(g(:));
% square structuring element for eroding the unknown region(s)
se = strel('square', 3);

%set a threshold for the minimum valid pixels in the neibourhood
%change the value here if the loop stucks
Nthres = 0;

n = 1;
unkreg = unkmask;
while n < nUnknown

    % get unknown pixels to process at this iteration
    unkreg = imerode(unkreg, se);
    unkpixels = ~unkreg&unkmask;
    [Y,X] = find(unkpixels); 
    
    for i = 1 : length(Y)
        
       
        % take current pixel
        x = X(i); 
        y = Y(i);
        c = reshape(im(y, x, :), [3, 1]);

        % take surrounding alpha values
        a = makewindow(y, x, N, alpha);
        
        % take surrounding foreground pixels
        f_pixels = makewindow(y, x, N, B);
        f_weights = (a .^ 2) .* g;
        f_pixels = reshape(f_pixels, N * N, 3);
        f_pixels = f_pixels(f_weights > 0, :);
        f_weights = f_weights(f_weights > 0);
        
        % take surrounding background pixels
        b_pixels = makewindow(y, x, N, B);
        b_weights = ((1 - a) .^ 2) .* g;
        b_pixels = reshape(b_pixels, N * N, 3);
        b_pixels = b_pixels(b_weights > 0, :);
        b_weights = b_weights(b_weights > 0);
        
        % if not enough data, return to it later...
        if length(f_weights) < Nthres || length(b_weights) < Nthres
            continue;
        end
        
        % partition foreground and background pixels to clusters (in a
        % weighted manner)
        [mu_f, Sigma_f] = cluster_OrachardBouman(f_pixels, f_weights, 0.05);
        [mu_b, Sigma_b] = cluster_OrachardBouman(b_pixels, b_weights, 0.05);


        
        % update covariances with camera variance, as mentioned in their
        % addendum
        Sigma_f = addCamVar(Sigma_f,0.01);
        Sigma_b = addCamVar(Sigma_b,0.01);
        
        % set initial alpha value to mean of surrounding pixels
        alpha_init = nanmean(a(:));
        
        % solve for current pixel
        [f, b, a] = solve1(mu_f, Sigma_f, mu_b, Sigma_b, c, 0.01, alpha_init, 50, 1e-6);
        F(y, x, :) = f;
        B(y, x, :) = b;
        alpha(y, x) = a;
        disp(a)
        unkmask(y, x)= 0; % remove from unkowns
        n = n + 1;
    end
end


function Sigma=addCamVar(Sigma,sigma_C)

Sigma=zeros(size(Sigma));
for i=1:size(Sigma,3)
    Sigma_i=Sigma(:,:,i);
    [U,S,V]=svd(Sigma_i);
    Sp=S+diag([sigma_C^2,sigma_C^2,sigma_C^2]);
    Sigma(:,:,i)=U*Sp*V';
end


