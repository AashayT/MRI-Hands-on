%Practicing with Cardiac MRI images
close All,
clear All,
clc

%STEP1: Loading the MRI image
%img = imread('mri_heart_1.jpg');
img = imread('Phantom1.png');
img = rgb2gray(img);
figure(1)
imshow(img)

%STEP2: Taking Fourier transform of MRI image
img_kspace = fftshift(fft2(img));
figure(2)
imshow(real(img_kspace))

%STEP3: Undersampling the k-space. 
%Rearrange the fft values before undersampling

    %US Scheme 1: We take alternate lines
    %und_samp_kspace = img_kspace;
    %und_samp_kspace(1:2:end,1:2:end)= 0;
    %figure(3)
    %imshow(real(und_samp_kspace))
    %title('50% undersampled')

    %US Scheme 2: Implementing high/low pass filter
%     undersampling_mask_LPF = ones(size(img_kspace,1), size(img_kspace,2));
%     undersampling_mask_LPF(140:284, 220:366)= 0;
%     und_samp_kspace = img_kspace - img_kspace.*undersampling_mask_LPF;
    
    %US Scheme 3: Random undersampling
%     rand_x_cord = randperm((size(img_kspace,1)*size(img_kspace,2)), 8000);
%     rand_x_cord = mod(rand_x_cord, size(img_kspace,1)) + 1;
%     rand_y_cord = randperm((size(img_kspace,1)*size(img_kspace,2)), 8000);
%     rand_y_cord = mod(rand_y_cord, size(img_kspace,2)) +1 ;
%     sample_points = [rand_x_cord' rand_y_cord'];
%     und_samp_temp = img_kspace;
%     for i=1:size(sample_points,1)
%        und_samp_temp(sample_points(i,1), sample_points(i,2)) = 0+0j;
%     end
%     und_samp_kspace = img_kspace-und_samp_temp;
    
    %US Scheme 4: Spiral undersampling
    dtheta = 5;
    W = 1/10000;
    theta = (2*pi/360).*[0:dtheta:floor(400/W)];
    spiral_x = ((W*360/2*pi).*(1.*theta).*cos(theta))';
    spiral_y = ((W*360/2*pi).*(1.*theta).*sin(theta))';
    spiral_points = floor([spiral_x spiral_y]);
    und_samp_temp = img_kspace;
    for i=1:size(spiral_points,1)
       if abs(spiral_points(i,1))< 0.5*(size(img_kspace, 1)) && abs(spiral_points(i,2))< 0.5*(size(img_kspace, 2))
           und_samp_temp(spiral_points(i,1) + floor(0.5*(size(img_kspace, 1))), spiral_points(i,2)+ floor(0.5*(size(img_kspace, 2)))) = 0+0j;
       end
    end
    und_samp_kspace = img_kspace-und_samp_temp;
    
    nonzero = sum(sum(und_samp_kspace ~= 0));
    factorUS = (nonzero/(size(img_kspace,1)*size(img_kspace,2))) ;
    
    figure(3)
    imshow(real(und_samp_kspace))
    title('percent undersampling')
    
%STEP4: Acquiring the image back
img_recov = ifft2(ifftshift(und_samp_kspace));
figure(4)
imshow(uint8(img_recov))
title('10% undersampled')

% %STEP5: Comparing images by taking histogram
% figure(5), histogram(img)
% figure(6), histogram(img_recov)
% 
% %STEP6: Taking wavelet transform of input MRI image

% For 1 level decomposition
% [cA, cH, cV, cD] = dwt2(img_recov,'db1');
% figure(7), imshow(cA)
% figure(8), imshow(cH)
% figure(9), imshow(cV)
% figure(10), imshow(cD)

% For N level decomposition
N = 2;
[C,S]= wavedec2(img,N,'db1');

% In the above [C] =  vector with all the coefficients of transform
% and [S] = contain structure of each transformed image

% Restructuring the coeff matrix [C] 
i = 1;
j = prod(S(1,:));
cA2 = reshape(C(i:j),S(1,:));

i = j+1;
j = i + prod(S(2,:)) - 1;
cH2 = reshape(C(i:j),S(2,:));

i = j+1; 
j = i + prod(S(2,:)) - 1;
cV2 = reshape(C(i:j),S(2,:));

i = j+1; 
j = i + prod(S(2,:)) - 1;
cD2 = reshape(C(i:j),S(2,:));

i = j+1; 
j = i + prod(S(3,:)) - 1;
cH1 = reshape(C(i:j),S(3,:));

i = j+1; 
j = i + prod(S(3,:)) - 1;
cV1 = reshape(C(i:j),S(3,:));

i = j+1; 
j = i + prod(S(3,:)) - 1;
cD1 = reshape(C(i:j),S(3,:));
% %     %See how the wavelet domain representation is different than fourier
% %     %domain represetation. MRI data is considerably sparse in wavelet domain.
% %     

% Displaying these images 
figure(11), imshow([[cA2,cH2(:,1:end-1); cV2(1:end-1,:),cD2(1:end-1,1:end-1)],cH1;cV1,cD1],'Colormap',pink)

% %Thresholding the transform domain
% t1= im2bw(uint8(real(cD)),0.3);
% t2= im2bw(uint8(imag(cD)),0.3);
% t = t1 + t2.*1i;
% 
% % %STEP7: Take inverse wavelet transform to recover the original image
% img_wav_recon = uint8(idwt2(cA, cH, cV, t, 'db1'));
% figure(11), imshow(img_wav_recon)






