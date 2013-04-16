function ainv=spai_s(A,S)
% >> INPUT
% A (required) : The coefficinet matrix.
% S (optional) : The sparsity pattern. The nonzero elements of ainv will
%                appear in the same position of the nonzero elements of S.
%                The default sparsity pattern is the same of A (S=A).

n=length(A);

% parameters
if(nargin==1 || isempty(S))
    S=A;
end

% sparsity pattern
[iid_all,jid_all]=find(S); % sparsity pattern of A

innz=length(iid_all);   % max number of nonzero entries in M
Mi=zeros(innz,1);       % row index of M
Mj=Mi;                  % col index of M
Ms=Mi;                  % values of M
Mnnz=0;                 % number of nonzero elements in M
for j = 1:n
    ej=sparse(j,1,1,n,1);
    jid=iid_all(jid_all==j);
    
    % adjusts A to the sparse-sparse mode
    [ss_A,ss_ej]=precond.sparse_sparse(jid,A,ej,[],[]);
    
    % solves the system (in sparse-sparse mode)
    ss_mj=ss_A\ss_ej;
    
    % updates the vectors of M
    nmj=length(jid);
    ind=Mnnz+1:Mnnz+nmj;
    Mi(ind)=jid;
    Mj(ind)=j;
    Ms(ind)=ss_mj;
    Mnnz=Mnnz+nmj;
end
% assembles the sparse matrix M
ind=1:Mnnz;
M=sparse(Mi(ind),Mj(ind),Ms(ind),n,n);

% set ainv
ainv.nnz=nnz(M);
ainv.apply=inline('M*x','M','x');
ainv.apply=@(x)ainv.apply(M,x);
ainv.M=M;
end