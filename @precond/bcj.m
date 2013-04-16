function ainv=bcj(A,droptol,nmax)
% Calculates the ainv given by Z*D*W, where D is diagonal matrix using the
% algorithm of biconjugation (See Benzi1999). 
%
% >> SYNTAX
% ainv = bcj(A,droptol) 
% 
% >> INPUT
% A       (required) : the coeff. matrix.
% DROPTOL (optional) : is the tolerance of drop strategy. Only the elements
%                      with value higher than DROPTOL are kept in factor
%                      matrices Z and W. The values of the matrix D are not
%                      droped.
% NMAX    (optional) : specifies the maximum number of nonzero elements in
%                      each column of the coef. matrix A. Just the biggest
%                      NMAX elements of Z and W that are bigger than
%                      droptol are kept. Its default value is
%                      NMAX=2*nnz(A)/n.
%
% >> OUTPUT
% AINV.M     : Z*D*W' (The ainv)
% AINV.APPLY : handle function of f(x)=Z*(D*(W*x))
% ainv.NNZ   : nnz(Z)+nnz(D)+nnz(W)

n=length(A);

% parameters
if(nargin<2 || isempty(droptol))
    droptol = 1E-1; % drop tolerance
end
if(nargin<3 || isempty(nmax))
    nmax=floor(2*nnz(A)/n);
end

% Collect the maximum value of A
[~,~,maxA]=find(A);
maxA=max(abs(maxA));

% Rescaling A
A=A/maxA;

% array of sparse vectors
W=cell(n,1);
Z=cell(n,1);
Q=cell(n,1);
P=cell(n,1);

d=nan(n,1);     % diagonal vector
nnzW=ones(n,1); % nnz of each sparse vector in W
nnzZ=ones(n,1); % nnz of each sparse vector in Z

% init as identity
for i = 1:n
    W{i}=sparse(i,1,1,n,1);
    Z{i}=W{i};
end

for i = 2:n
    zi=Z{i};
    wi=W{i};
    
    % update Q, P and d
    Q{i-1}=(W{i-1}')*A;
    P{i-1}=A*Z{i-1};
    d(i-1)=(W{i-1}')*P{i-1};
    
    for j=1:(i-1)
        wi=wi-(Q{j}(i)/d(j))*W{j};
        zi=zi-(P{j}(i)/d(j))*Z{j};
    end
    
    % DROP: Just the biggest NMAX elements of Z and W that are bigger than
    % droptol are kept.
    abs_wi=abs(wi);
    abs_zi=abs(zi);
    sorted_wi=sort(abs_wi,'descend');
    sorted_zi=sort(abs_zi,'descend');
    drop_wi=max(sorted_wi(nmax),droptol);
    drop_zi=max(sorted_zi(nmax),droptol);
    wi(abs_wi<drop_wi)=0;
    zi(abs_zi<drop_zi)=0;
    
    % update Z and W
    Z{i}=zi;
    W{i}=wi;    
    
    nnzZ(i)=nnz(zi);
    nnzW(i)=nnz(wi);
end
d(n)=(wi')*A*zi;
D=sparse(1:n,1:n,1./d,n,n);

% build the matrices W and Z
zi=zeros(sum(nnzZ),1);zj=zi;zs=zi;nZ=0;
wi=zeros(sum(nnzW),1);wj=wi;ws=wi;nW=0;
for i = 1:n
    indZ=(nZ+1):(nZ+nnzZ(i));
    indW=(nW+1):(nW+nnzW(i));
    [zi(indZ),zj(indZ),zs(indZ)] = find(Z{i});
    [wi(indW),wj(indW),ws(indW)] = find(W{i});
    zj(indZ)=zj(indZ)+(i - 1);
    wj(indW)=wj(indW)+(i - 1);
    nZ=nZ+nnzZ(i);
    nW=nW+nnzW(i);
end
W=sparse(wi,wj,ws,n,n)';
Z=sparse(zi,zj,zs,n,n);

% set ainv
ainv.nnz = nnz(Z)+nnz(D)+nnz(W);
ainv.apply=inline('Z*(D*(W*x))','Z','D','W','x');
ainv.apply=@(x)ainv.apply(Z,D,W,x);
ainv.M=(Z*D*W)/maxA;