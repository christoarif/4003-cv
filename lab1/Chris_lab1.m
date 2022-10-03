%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.1a) input imag
Pc = imread('mrt-train.jpeg');
whos Pc
P = rgb2gray(Pc); 

%% 2.1b) view image 
imshow(P)

%% 2.1c) check min and max intensities
minP = min(P(:)); %min is 13
maxP = max(P(:)); %max is 204

%% 2.1d) contrast stretching. min should be 0 and max should be 255
Pstretch = imsubtract(P, double(minP));
P2 = immultiply(Pstretch, (255 ./ double(maxP-minP)));
minP2 = min(P2(:)); %should be 0 
maxP2 = max(P2(:)); %should be 255
%% 2.1e) redisplay image after contrast strectching
imshow(P2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.2 Histogram equalization
%% 2.2a) Display intensity with 10 bins and 256 bins

imhist(P,10);
%%
imhist(P,256);

%% 2.2b) Histogram Equalization and Redisplay with 10 bins, 255 bins

P3 = histeq(P,255);
%% 
imhist(P3,10);
%%
imhist(P3, 256);

%% 2.2c) Rerun histeq on P3

P3_new = histeq(P3,255);
imhist(P3_new, 256)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.3 Linear Spatial Filtering 
%% 2.3a i ) 5x5 with sigma 1 
kernel = -2:2; %X and Y dimension is both 5 
sigma_part1 = 1.0;

[X,Y] = meshgrid(kernel);

h_part1 = (1 / (2 * pi * sigma_part1)) * exp(-((X).^2 + (Y).^2) / (2 * sigma_part1^2)); %the formula

mesh(h_part1);

%% 2.3a ii ) 5x5 with sigma 2
kernel = -2:2;
sigma_part2 = 2.0;

[X,Y] = meshgrid(kernel);

h_part2 = (1 / (2 * pi * sigma_part2)) * exp(-((X).^2 + (Y).^2) / (2 * sigma_part2^2));

mesh(h_part2);
%% 2.3b) picture with gaussian noise
ntu_gn = imread('ntugn.jpeg');
imshow(ntu_gn)
%% 2.3c) filter the gaussian noise picture with the filters
ntu_gn_h1 = conv2(ntu_gn, h_part1);
imshow(ntu_gn_h1, []); %can also convert the ntu_gn_h1 to uint8 first. 
%%
ntu_gn_h2 = uint8(conv2(ntu_gn, h_part2));
imshow(ntu_gn_h2); %ntu_gn_h2 was converted to uint8
%% 2.3d) picture with speckle noise
ntu_sp = imread('ntusp.jpeg');
imshow(ntu_sp)
%% 2.3e) filter the speckle picture with the filters
ntu_sp_h1 = conv2(ntu_sp, h_part1);
imshow(ntu_sp_h1, []); %can also convert the ntu_gn_h1 to uint8 first. 
%% 
ntu_sp_h2 = uint8(conv2(ntu_sp, h_part2));
imshow(ntu_sp_h2); %ntu_gn_h2 is converted to uint8

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.4 Median Filtering
ntu_gn_medfilt3 = medfilt2(ntu_gn); %default is 3 by 3 
ntu_gn_medfilt5 = medfilt2(ntu_gn, [5 5]);

ntu_sp_medfilt3 = medfilt2(ntu_sp); %default is 3 by 3 
ntu_sp_medfilt5 = medfilt2(ntu_sp, [5 5]);
%%
imshow(ntu_gn_medfilt3)
%%
imshow(ntu_gn_medfilt5)
%%
imshow(ntu_sp_medfilt3)
%%
imshow(ntu_sp_medfilt5)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2.5 Supressing Noise Interference Patterns
%% 2.5a) download noisy pck image
pckint = imread('pckint.jpeg');
imshow(pckint)
%% 2.5b) obtain ft of the image and compute the power spectrum S
F = fft2(pckint); %F is complex matrix
S = abs(F).^2; %make S to be real matrix
%% 
imagesc(fftshift(S.^0.1));
colormap('default');
%% 2.5c) redisplay S without fftshift
imagesc(S.^0.1);
% actual locations of peaks : (9,241), (249,17)
%% 2.5d) set the 5x5 neighbourhood of the peaks to be 0
F(241-2:241+2, 9-2:9+2) = 0;
F( 17-2:17+2, 249-2:249+2) = 0;
%% recompute power spectrum and display
S = abs(F).^2;
imagesc(fftshift(S.^0.1));
colormap('default');

%% 2.5e) find inverse ft and display
inverse_F = uint8(ifft2(F));
imshow(inverse_F);

%% TODO how to improve current result
% the contrast of the image decrease after doing the ft noise cancellation,
% so we can imrpove the quality by contrast stretching.
%% 2.5f) freeing the primate by filtering out the fence.
primate = imread('primatecaged.jpeg');
primate_grey = rgb2gray(primate); 
imshow(primate_grey)
%%
%start with same approach
F_primate = fft2(primate_grey);
S_primate = abs(F_primate).^2;
imagesc((S_primate.^0.01));
colormap('default');

%% 
% peaks identified: (253, 11) , ( 6,247), (248, 21), (10, 236)
F_primate(253-2:253+2, 11-2:11+2 ) = 0;
F_primate(6-2:6+2, 247-2:247+2) = 0;
F_primate(248-2:248+2,21-2:21+2 ) = 0;
F_primate(10-2:10+2, 236-2:236+2) = 0;

%%
S_primate = abs(F_primate).^2;
imagesc((S_primate.^0.1));
colormap('default');
%%
inverse_F_primate = uint8(ifft2(F_primate));
imshow(inverse_F_primate);

%% 2.6 Undoing Perspective Distortion of Planar Surface
%2.6.a
book_img = imread('book.jpeg');
imshow(book_img)

%% 2.6.b

[X Y] = ginput(4);
%%
x_desired = [0 0 210 210]; % 1 pixel to 1 mm of the book dimension, as stated in the qn
y_desired = [0 297 297 0];

%% 2.6c) set up matrices required to estimate the projective transfomration

v=[0;0;0;297;210;297;210;0] ;

A = [ X(1) Y(1) 1 0 0 0 -x_desired(1)*X(1) -x_desired(1)*Y(1);
    0 0 0 X(1) Y(1) 1 -y_desired(1)*X(1) -y_desired(1)*Y(1);
    X(2) Y(2) 1 0 0 0 -x_desired(2)*X(2) -x_desired(2)*Y(2);
    0 0 0 X(2) Y(2) 1 -y_desired(2)*X(2) -y_desired(2)*Y(2);
    X(3) Y(3) 1 0 0 0 -x_desired(3)*X(3) -x_desired(3)*Y(3);
    0 0 0 X(3) Y(3) 1 -y_desired(3)*X(3) -y_desired(3)*Y(3); 
    X(4) Y(4) 1 0 0 0 -x_desired(4)*X(4) -x_desired(4)*Y(4);
    0 0 0 X(4) Y(4) 1 -y_desired(4)*X(4) -y_desired(4)*Y(4); ]


u = A \ v;

U = reshape([u;1], 3,3)';
w = U*[X'; Y'; ones(1,4)]; 
w = w ./ (ones(3,1) * w(3,:))

%% 2.6d) Warp the image

T = maketform('projective', U');

P2 = imtransform(book_img, T, 'XData', [0 210], 'YData', [297 0]);

%% 2.6e) Display image
imshow(P2)

