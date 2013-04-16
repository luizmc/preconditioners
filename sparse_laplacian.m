function A = sparse_laplacian (nxyz,dxyz,cxyz,K)
%
% Generates a 2D or 3D sparse discrete Laplacian matrix.
%
% Input:
%    nxyz = [nx,ny] or [nx,ny,nz] (for 2D or 3D problems)
%    dxyz = [dx,dy] or [dx,dy,dz] or []
%           defaults to 1.0, in the latter case
%    cxyz = [cx,cy] or [cx,cy,cz] or []
%           anisotropy coefficient
%           defaults to 1.0, in the latter case
%    K    = [matrix of permeabilities] or []
%           defaults to matrix of 1's
%           If informed, must be nx-by-ny.
%           (Permeability assumed not to vary along z-direction.)
%
% Output:
%    A    = sparse matrix, (nx*ny)-by-(nx*ny) or (nx*ny*nz)-by-(nx*ny*nz)
%           natural ordering, x varies first, z varies last.

   nx = nxyz(1);
   ny = nxyz(2);
   if length(nxyz)==3,
      nz = nxyz(3);
   else
      nz = 1;
   end      
   
   if isempty(dxyz),
      dx = 1.0;
      dy = 1.0;
      dz = 1.0;
   else
      dx = dxyz(1);
      dy = dxyz(2);
      if length(dxyz)==3,
         dz = dxyz(3);
      else
         dz = 1.0;
      end        
   end

   if isempty(cxyz),
      coefx = 1.0;
      coefy = 1.0;
      coefz = 1.0;
   else
      coefx = cxyz(1);
      coefy = cxyz(2);
      if length(cxyz)==3,
         coefz = cxyz(3);
      else
         coefz = 1.0;
      end        
   end

   if isempty(K),
      A = sparse_laplacian_without_K (nx, ny, nz, dx, dy, dz, coefx, coefy, coefz);
   else
      A = sparse_laplacian_with_K (K, nx, ny, nz, dx, dy, dz, coefx, coefy, coefz);
   end   

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = sparse_laplacian_without_K (nx, ny, nz, dx, dy, dz, coefx, coefy, coefz)

   % A assumes the ordering in which x varies first, y second and z last.
   
   if nz == 1,
      is3D = 0;
   else
      is3D = 1;
   end
   
   I_nx = speye(nx,nx);
   I_ny = speye(ny,ny);
   I_nz = speye(nz,nz);
   lap1d_nx = (coefx/(dx^2)) * my_lap1d(nx);
   lap1d_ny = (coefy/(dy^2)) * my_lap1d(ny);
   lap1d_nz = (coefz/(dz^2)) * my_lap1d(nz);
   A = kron(I_nz,kron(I_ny,lap1d_nx)) + ...
       kron(I_nz,kron(lap1d_ny,I_nx)) + ...
       kron(lap1d_nz,kron(I_ny,I_nx))*is3D;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function M = my_lap1d(n)
   if n>1,
      M = spdiags(ones(n,1)*[1,-2,1],[-1,0,1],n,n); 
   else
      M = sparse(-2);
   end   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = sparse_laplacian_with_K (K, nx, ny, nz, dx, dy, dz, coefx, coefy, coefz)

   % K is 2D (directions x and y)
   % if nz>1, K is repeated at each layer
   %
   % A assumes the ordering in which x varies first, y second and z last.

   if ((size(K,1)~=nx)||(size(K,2))~=ny),
      error ('K must be nx-by-ny!!!')
   end

   % numbering
   numbering = (1:nx)'*ones(1,ny) + nx*ones(nx,1)*(0:(ny-1));

   % x-interfaces
      
   % harmonic averages
   Kx = (coefx/dx^2)*2*[K(1,:);K].*[K;K(end,:)]./([K(1,:);K]+[K;K(end,:)]);
   
   d = -reshape(Kx(1:(end-1),:)+Kx(2:end,:),[],1);
   A = spdiags(d,0,nx*ny,nx*ny);
   
   i = reshape([numbering(1:(end-1),:);numbering(2:end,:)],[],1);
   j = reshape([numbering(2:end,:);numbering(1:(end-1),:)],[],1);
   v = reshape(kron([1;1],Kx(2:(end-1),:)),[],1);

   A = A + sparse(i,j,v,nx*ny,nx*ny);
   
   
   
   Ky = (coefy/dy^2)*2*[K(:,1),K].*[K,K(:,end)]./([K(:,1),K]+[K,K(:,end)]);
   
   d = -reshape(Ky(:,1:(end-1))+Ky(:,2:end),[],1);
   A = A + spdiags(d,0,nx*ny,nx*ny);
   
   i = reshape([numbering(:,1:(end-1)),numbering(:,2:end)],[],1);
   j = reshape([numbering(:,2:end),numbering(:,1:(end-1))],[],1);
   v = reshape(kron([1,1],Ky(:,2:(end-1))),[],1);

   A = A + sparse(i,j,v,nx*ny,nx*ny);
   
   if nz > 1,
      A = kron(speye(nz,nz),A);
      Kz = (coefz/dz^2)*K;
	   A = A + kron(spdiags(ones(nz,1)*[1,-2,1],[-1,0,1],nz,nz),spdiags(reshape(Kz,[],1),0,nx*ny,nx*ny));
   end
end
   