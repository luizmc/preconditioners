function ainv=mr_a(A)
n=length(A);
% NMAX (optional) : max number of nonzero on each column of ainv.

% set the parameters
nlocal=20;
tol=1E-2;
innz=min(5,n); % max number of nonzeros per column

% initializes M
scale=trace(A)/linalg.innerprod(A,A); % min ||I-A*(alpha*I)||_F
M=scale*speye(n);

norm_Aek=-ones(n,1);
for j = 1:n     % set mj
    ej=sparse(j,1,1,n,1);
    mj=sparse(j,1,scale,n,1);
    rj=ej-A*mj;
    rj2=norm(rj)^2;
    jid=j; % just the diagonal
    
    % updates the set jid (full mode)
    [jnew,norm_Aek]=...
        precond.jnew_adaptative(A,norm_Aek,rj,rj2,innz,jid);
    jid=sort([jid;jnew]);
    
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