function ainv=mr_s(A,S)
% >> INPUT
% A (required) : The coefficinet matrix.
% S (optional) : The sparsity pattern. The nonzero elements of ainv will
%                appear in the same position of the nonzero elements of S.
%                The default sparsity pattern is the same of A (S=A).

n=length(A);

if(nargin<2 || isempty(S))
    S=A^2;
end

% set the parameters
nlocal=5;
tol=1E-2;

% initializes M
scale=trace(A)/linalg.innerprod(A,A); % min ||I-A*(alpha*I)||_F
M=scale*speye(n);

[iid_all,jid_all]=find(S); % sparsity pattern of A
for j = 1:n     % set mj
    ej=sparse(j,1,1,n,1);
    mj=sparse(j,1,scale,n,1);    
    jid=unique(iid_all(jid_all==j)); % sparsity pattern of Mj
    
    % adjusts A to the sparse-sparse mode
    [ss_A,ss_ej,mj]=precond.sparse_sparse(jid,A,ej,mj,[]);
    
    [nrow,ncol]=size(ss_A);
    % solves Amj=ej (sparse-sparse)
    if(nrow==ncol) % square system :: minimal residue
        for k = 1:nlocal
            ss_rj=ss_ej-ss_A*mj;
            ss_rj2=norm(ss_rj)^2;
            if(ss_rj2 < tol)
                break;
            end
            Arj=ss_A*ss_rj;
            alpha=(ss_rj'*Arj)/norm(Arj)^2;
            mj=mj+alpha*ss_rj;
        end
    else % rectangular system :: steepest descent
        for k = 1:nlocal
            ss_rj=ss_ej-ss_A*mj;
            ss_rj2=norm(ss_rj)^2;
            if(ss_rj2 < tol)
                break;
            end            
            
            gj=ss_A'*ss_rj;
            Agj=ss_A*gj;
            alpha=(norm(gj)/norm(Agj))^2;
            mj=mj+alpha*gj;
        end
    end
    M(jid,j)=mj; % updates M    
end

% set ainv
ainv.nnz=nnz(M);
ainv.apply=inline('M*x','M','x');
ainv.apply=@(x)ainv.apply(M,x);
ainv.M=M;
end