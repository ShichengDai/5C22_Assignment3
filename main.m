im=imread('images/gandalf/input-small.png');
trimap=imread('images/gandalf/trimap.png');

im = im2double(im);
trimap = im2double(trimap);

[F,B,alpha]=bayesmat(im,trimap);
