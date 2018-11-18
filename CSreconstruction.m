% Energy function for compressed sensing reconstruction
% E = || F(desired_img) - y ||.^2 + lambda||grad(desired_img)||

function [new_filtered_image] = CSreconstruction(und_samp_kspace)
    
    lambda = 0.1;
    eps = 1e-6;
    alpha = 0.01;
    [ny nx] = size(img_recov);
    [Grad,Dx,Dy] = spmat_gradient2d(ny,nx,1);
    
    % Initializing the filtered image
    %acquired_image = real(ifft2(ifftshift(und_samp_kspace)));
    acquired_image = double(real(img_recov));
    figure(1), imshow(uint8(acquired_image))
    
    % Generating copy for beginning of optimization
    filtered_image = acquired_image;
    %figure(), imshow(uint8(filtered_image))

    for i=1:100
    %% Calculating the energy functionals

        % Using L2-Norm for calculating data fidelity term. Considering that the
        % errors between actual and acquired pixel values follow a Gaussian
        % distribution
        E1 = 2*(filtered_image - acquired_image).^2;

        % Using L1-Norm for the data smoothness term. L1 norm or Thikhonov
        % regularizer uses total variation of the filtered image to penalize the
        % sudden discontinuities. L1-Norm is an edge preverving regularizer
        [grad_filtered_x, grad_filtered_y] = gradient(filtered_image);
        Norm_grad = sqrt(grad_filtered_x.^2 + grad_filtered_y.^2) + eps;
        E2 = abs(Norm_grad);
        %figure(), imshow(uint8(Norm_grad))

        TE = sum(sum(E1 + lambda*E2));
        figure(2), plot(i,TE, 'o'), hold on

        %% Calculation of Getaux derivative or directional derivative of T.E.
        L1 = 2*abs(filtered_image - acquired_image);

        [X Y] = meshgrid(1:size(filtered_image,2), 1:size(filtered_image,1));
        X = X/size(filtered_image,2);
        Y = Y/size(filtered_image,1);
        L2 = divergence(X, Y, (grad_filtered_x./Norm_grad), (grad_filtered_y./Norm_grad));

        dL_du = L1 - lambda*L2;
        
        %sp_dL_du = spdiags(dL_du(:),0,(size(dL_du,1)*size(dL_du,2)),(size(dL_du,1)*size(dL_du,2)));
        %A = pcg(sp_dL_du, acquired_image(:));
        
        figure(3), plot(i, sum(sum(dL_du)),'o'), hold on
        
        filtered_image = filtered_image - alpha*dL_du;
        %figure(3), imshow(uint8(filtered_image)), hold on;
        if mod(i,5)==0
            Norm_gragient{i/5} = Norm_grad;
            new_filtered_image{i/5} = filtered_image;
        end
    end
end