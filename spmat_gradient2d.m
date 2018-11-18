function [grad,dx,dy] = spmat_gradient2d(ny, nx, nc)
%spmat_gradient2d Assembles linear operator for gradient
%   Input args are nx, ny, the dimension of the image and nc the number of
%   color channels. The resulting gradient operates on a column wise 
%   stacked image vector with the images from the individual color channels
%   being stacked as well, i.e.
%       [ f_c1 ; f_c2 ; ... ]
%   where f_c1 is the stacked image of the first color channel.
%   The vector resulting from applying this gradient operator is of shape
%       [ dx_c1 ; dx_c2 ; ... ; dy_c1 ; dy_c2; ... ]
%   where dx_c1 is the R^(ny*nx) vector that is the x gradient for the
%   first color channel. Therefore the output is a (2*ny*nx*nc)x(ny*nx) 
%   sparse matrix for the gradient, and (ny*nx*nc)*(ny*nx) sparse matrices 
%   for dx and dy.

    dy = spdiags([[-ones(ny - 1, 1); 0], ones(ny, 1)], [0, 1], ny, ny);
    dy = kron(speye(nx), dy);
    
    dx = spdiags([[-ones(ny*(nx-1),1); zeros(ny, 1)], ones(nx*ny,1)], ...
                 [0, ny], nx*ny,nx*ny);
    
    grad = cat(1, kron(speye(nc), dx), kron(speye(nc), dy));
end
