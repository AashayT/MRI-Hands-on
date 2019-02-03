% Energy function for compressed sensing reconstruction
% E = || F(desired_img) - y ||.^2 + lambda||phi(desired_img)||
% Type of function 'phi' is chosen from the specific application. Usually a
% function phi is chosen such that it transforms the image into a sparse
% domain.
% For angiography images: Gradient
%     Normal MRI images : Wavelet 
%
% Currently only Total variation is considered as the sparsifying transform
% TODO: Addition of wavelet into the reconstruction equation
%
% function [new_filtered_image] = CSreconstruction(und_samp_kspace)
% 
% where     new_filtered_image = a list with images stored at interval of 5
%           und_same_kspace = k-space undersampled data

function [new_filtered_image] = CSreconstruction(und_samp_kspace)
    %close all
    
    % Parameter initialization
    % lambda = regularizer weighting between consistency and smoothness
    %           term
    % alpha = step size for gradient decent algorithm
    lambda = 0.1;
    delta = 10; 
    eps = 1e-6;
    alpha = 0.05;
    [ny nx] = size(und_samp_kspace);
    [Grad,Dx,Dy] = spmat_gradient2d(ny,nx,1);
    
    %For testing on noisy image in k-space representation
    acquired_image = real(ifft2(ifftshift(und_samp_kspace)));
    acquired_image = double(uint8(acquired_image));
    
    %For testing on noisy image in pixel representation
    %acquired_image = double(uint8(real(und_samp_kspace)));
    %figure(1), imagesc(acquired_image), colormap gray
    
    % Initializing the filtered image
    % Here we have considered the 1st filtered image = acquired image.
    % However this can be initialized differently 
    
    filtered_image = 1*acquired_image;
    %figure(), imagesc(filtered_image), colormap gray
    
    % Specifying the inhomogenous Diffusion matrix
    [grad_acquired_x, grad_acquired_y] = gradient(acquired_image);
    Norm_grad_acquired = sqrt(grad_acquired_x.^2 + grad_acquired_y.^2) + eps;
    D = DiffusionCoeff(Norm_grad_acquired);
    
    %Iterations are hard coded, can be provided as function argument
    for i=1:500
    
        %% Smoothening the boundaries
        %if mod(i,100) == 0
        %    [grad_Dx, grad_Dy] = gradient(D);
        %    grad_Dtotal = sqrt(grad_Dx.^2 + grad_Dy.^2);
        %    D = D + grad_Dtotal;
        %end            

    %% Calculating the energy functionals

        % Using L2-Norm for calculating data fidelity term. Considering that the
        % errors between actual and acquired pixel values follow a Gaussian
        % distribution
        E1 = 1*((filtered_image - acquired_image).^2);
        % Initially both images are same. Therefore 
        % max(max(E1))== min(min(E1)) = 0
        
        %ISSUE: The data fiedelity term causes the optimization to explode
        %Everything works fine when only smoothness term is considered in
        %the optimization.
        %Assigning constant of 0.1 to data fidelity term in Energy eq. 
        %works best. But more research is needed 

        % Using L1-Norm for the data smoothness term. L1 norm or Thikhonov
        % regularizer uses total variation of the filtered image to penalize the
        % sudden discontinuities. L1-Norm is an edge preverving regularizer
        [grad_filtered_x, grad_filtered_y] = gradient(filtered_image);
        Norm_grad = sqrt(grad_filtered_x.^2 + grad_filtered_y.^2) + eps;
        
        % Rescaling of Intensities
        Scaling_factor_grad = max(max(acquired_image))/max(max(Norm_grad));
        %Scaling_factor_grad = 1;
        Norm_grad_scaled = Scaling_factor_grad*Norm_grad;
        
        E2 = abs(D.*Norm_grad_scaled);
        %E2 = Norm_grad.^2;
        %figure(), imagesc(Norm_grad)
        
        %Adding curve length
        [grad_Dx, grad_Dy] = gradient(D);
        grad_Dtotal = sqrt(grad_Dx.^2 + grad_Dy.^2);
        E3 = abs(grad_Dtotal);

        TE = sum(sum(0.001*E1 + lambda*E2 + delta*E3));
        figure(2), plot(i,TE, 'o'), hold on
        
        %% Calculation of Getaux derivative or directional derivative of T.E.
        L1 = 1*abs(filtered_image - acquired_image);

        [X, Y] = meshgrid(1:size(filtered_image,2), size(filtered_image,1):1);
        
        %Whether to scale it or not, needs to be investigated further
        %X = X/size(filtered_image,2);
        %Y = Y/size(filtered_image,1);
        Div = divergence(X, Y, (D.*grad_filtered_x./(Norm_grad)), (D.*grad_filtered_y./(Norm_grad)));
        %L2 = divergence(X, Y, (grad_filtered_x), (grad_filtered_y));
        
        %Scaling for divergence
        Scaling_factor_div = max(max(acquired_image))/max(max(Div));
        %Scaling_factor_div = 1;
        Div_scaled = Scaling_factor_div*Div;
        L2 = Div_scaled;
        
        L3 = divergence(X, Y, (grad_Dx), (grad_Dy));
        
        
        dL_du = 0.001*L1 - lambda*L2 - delta*L3;
        
        figure(3), plot(i,sum(sum(dL_du)), 'x'), hold on
        
        
        %Writing the solution in form of a linear equation AX=B. 
        %Following formulation may not be perfect. Please verify
        %sp_dL_du = spdiags(dL_du(:),0,(size(dL_du,1)*size(dL_du,2)),(size(dL_du,1)*size(dL_du,2)));
        %A = pcg(sp_dL_du, acquired_image(:));
        %filtered_image = A;
        
        %figure(3), plot(i, sum(sum(dL_du)),'o'), hold on
        
        filtered_image = filtered_image - alpha*(dL_du);
        %figure(3), imagesc(filtered_image), colormap gray, hold on;
        
        %Saving the denoised/reconstructed image in a folder. 
        %Images are saved at interval of 5
        if mod(i,5)==0
            Norm_gradient{i/5} = Norm_grad_scaled;
            Divergence{i/5} = Div_scaled;
            new_filtered_image{i/5} = filtered_image;
            imwrite(uint8(new_filtered_image{i/5}),['./temp/Intensities/I' num2str(i)], 'png');
            imwrite(uint8(Norm_gradient{i/5}),['./temp/Gradients/Grad' num2str(i)], 'png');
            imwrite(uint8(Divergence{i/5}),['./temp/Divergenge/Div' num2str(i)], 'png');
        end
        
        if mod(i,50) == 0
            imwrite(uint8(255.*D),['./temp/D/D' num2str(i)], 'png');
        end
    end
    
% N = size(img_recov,1)*size(img_recov,2);
% [j] = [0:N-1]';
% omega = exp(-2*pi*1i/N);
% DFT_mat = (omega.^(j.*j))./sqrt(N);
% DFT_mat_sparse = spdiags(DFT_mat(:), 0, N, N);
% 
% Undersampling_matrix = (und_samp_kspace ~= 0);
% Undersampling_matrix_flat = reshape(Undersampling_matrix, [size(img,1)*size(img,2), 1]);
% US_mat_sparse = spdiags(Undersampling_matrix_flat(:), 0, N, N);
% 
% 
% img_sparse = spdiags(img(:), 0, N, N);
% test = DFT_mat_sparse*img_sparse;
% DFT_img = reshape(full(diag(test)), [size(img,1), size(img,2)]);


    
