% Energy function for compressed sensing reconstruction
% E = || F(desired_img) - y ||.^2 + lambda||grad(desired_img)||

% Initializing the filtered image
acquired_image = real(ifft2(ifftshift(und_samp_kspace)));
lambda = 0.1;

% Generating copy for beginning of optimization
filtered_image = acquired_image;
%figure(), imshow(uint8(filtered_image))

for i=1:100
%% Calculating the energy functionals

    % Using L2-Norm for calculating data fidelity term. Considering that the
    % errors between actual and acquired pixel values follow a Gaussian
    % distribution
    E1 = (filtered_image - acquired_image).^2;

    % Using L1-Norm for the data smoothness term. L1 norm or Thikhonov
    % regularizer uses total variation of the filtered image to penalize the
    % sudden discontinuities. L1-Norm is an edge preverving regularizer
    [grad_filtered_x, grad_filtered_y] = gradient(filtered_image);
    E2 = lambda*(grad_filtered_x + grad_filtered_y);

    TE = sum(sum(E1 + E2));

    %% Calculation of Getaux derivative or directional derivative of T.E.
    L1 = 2*abs(filtered_image - acquired_image);

    [X Y] = meshgrid(1:size(filtered_image,2), 1:size(filtered_image,1));
    Norm_grad = sqrt(grad_filtered_x.^2 + grad_filtered_y.^2);
    L2 = divergence(X, Y, (grad_filtered_x./Norm_grad), (grad_filtered_y./Norm_grad));

    dL_dt = L1 - L2;

    filtered_image = filtered_image - dL_dt;
    if mod(i,5)==0
        new_filtered_image{i/5} = filtered_image;
    end
end