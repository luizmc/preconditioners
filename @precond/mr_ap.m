function ainv=mr_ap(A)
warning('PRECOND:UNCHECKED','The implementation of mr_ap precond is wrong.');

n=length(A);

% set the parameters
nlocal=20;
tol=1E-3;
innz=min(5,n); % max number of nonzeros per column

alpha=trace(A)/linalg.innerprod(A,A); % min ||I-A*(alpha*I)||_F
M=alpha*speye(n);

norm_Aek=-ones(n,1);
for j = 1:n	% set mj
    ej=sparse(j,1,1,n,1);
    mj=M(:,j);
    rj=ej-A*mj;
    rj2=norm(rj)^2;
    jid=j;
    
    % updates the set jid (full mode)
    [jnew,norm_Aek]=...
        precond.jnew_adaptative(A,norm_Aek,rj,rj2,innz,jid);
    jid=sort([jid;jnew]);
    
    % adjusts A to the sparse-sparse mode
    [ss_A,ss_ej,mj,ss_M]=precond.sparse_sparse(jid,A,ej,mj,M);
    
    [nrow,ncol]=size(ss_A);
    % solves Amj=ej (sparse-sparse)
    if(nrow==ncol) % square system :: minimal residue
        for k = 1:nlocal
            ss_rj=ss_ej-ss_A*mj;
            ss_rj2=norm(ss_rj)^2;
            if(ss_rj2 < tol)
                break;
            end
            zj=ss_M*ss_rj;
            qj=ss_A*zj;
            alpha=(ss_rj'*qj)/(norm(qj)^2);
            mj=mj+alpha*zj;
        end
    else % rectangular system :: gradient
        for k = 1:nlocal
            ss_rj=ss_ej-ss_A*mj;
            ss_rj2=norm(ss_rj)^2;
            if(ss_rj2 < tol)
                break;
            end
            gj=ss_A'*(ss_M'*(ss_M*ss_rj));
            Agj=ss_A*gj;
            alpha=(norm(gj)/norm(Agj))^2;
            mj=mj+alpha*gj;
        end
    end
    
    M(jid,j)=mj;
end
% set ainv
ainv.nnz=nnz(M);
ainv.apply=inline('M*x','M','x');
ainv.apply=@(x)ainv.apply(M,x);
ainv.M=M;
end