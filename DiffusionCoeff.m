
function [Diffusion_coeff_matrix] = DiffusionCoeff(Norm_grad_scaled)
    
    [ny, nx] = size(Norm_grad_scaled);
    mean_image = mode(reshape(Norm_grad_scaled, [nx*ny, 1]));
    var_image = var(reshape(Norm_grad_scaled, [nx*ny, 1]));
    sd_image = sqrt(var_image);
    threshold = mean_image + sd_image;
    %Diffusion_coeff_matrix = Norm_grad_scaled < threshold;
    Diffusion_coeff_matrix = 1-1./(1+exp(-(Norm_grad_scaled - ...
                                        threshold)./(0.1*threshold)));
    Diffusion_coeff_matrix(1,:) = 0;
    Diffusion_coeff_matrix(ny,:) = 0;
    Diffusion_coeff_matrix(:, 1) = 0;
    Diffusion_coeff_matrix(:, nx) = 0;
    

end



