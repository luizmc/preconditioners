function G=fsai(A)
warning('PRECOND:UNCHECKED','The implementation of mr_ap precond is wrong.');

% The preconditioned matrix is given by G*A*G'

n=length(A);

% parameters
k=1;

[iid_all,jid_all]=find(tril(A^k)); % sparsity pattern of A^k 

M=cell(n,1);
nnzG=zeros(n,1);

for j = 1:n
    jid=iid_all(jid_all==j);
    ej=sparse(j,1,1,n,1);
    
    [ss_A,ss_ej]=ainv.sparse_sparse(jid,A,ej,[],[]);
    
    % solves the system (in sparse-sparse mode)
    ss_mj=ss_A\ss_ej; % least-square
    
    % update M
    M{j}=sparse(jid,ones(size(jid)),ss_mj,n,1);
    nnzG(j)=length(ss_mj);
end

% assembles M
nM=sum(nnzG);
i=zeros(nM,1);
j=i;
s=i;
nM=0;
for k = 1:n
    ind=(nM+1):(nM+nnzG(k));
    [i(ind),j(ind),s(ind)] = find(M{k});    
    j(ind)=j(ind)+(k - 1);
    nM=nM+nnzG(k);
end
M = sparse(i,j,s,n,n);

D = sqrt(1./diag(M));ind=1:n;
D = sparse(ind,ind,D,n,n); % D=(diag(G))^{-1/2}
G = M*D;
end