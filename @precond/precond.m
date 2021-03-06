classdef precond
    methods(Static)                
        % spai (I = A*M)
        ainv=spai_a(A) % adapt  sparse pattern
        ainv=spai_s(A) % static sparse pattern
        
        % minimal residue (I = A*M)
        ainv=mr_s(A)   % static sparse pattern
        ainv=mr_a(A)   % adapt  sparse pattern
        ainv=mr_ap(A)  % adapt  sparse pattern, self-preconditioning
        
        % factorized sparse approximated inverse (I=M'*A*M)
        ainv=fsai(A)
        
        % biconjugation (I=A*Z*D*W')
        ainv=bcj(A)
        
        function M = spai(A)
            M=spai(A);
        end
        
        function R = ichol_nofill(A)
            % Produces the incomplete Cholesky factor of a real sparse
            % matrix that is symmetric and positive definite using no
            % fill-in.
            
            % check the symmetry
            if(normest(A-A') > 1E-8)
                error('ICHOL:INVALIDINPUT','The matrix must be positive definite');
            end
            
            if(exist('ichol','builtin') > 0)
                opts.type='nofill';
                R=ichol(A,opts);
            else
                R=cholinc(A,'0');
            end
        end                
        
        function R = ichol_ict(A,droptol)
            % Performs the incomplete Cholesky factorization of X, with
            % drop tolerance droptol
            
            % check the symmetry
            if(normest(A-A') > 1E-8)
                error('ICHOL:INVALIDINPUT','The matrix must be positive definite');
            end
            
            if(nargin<2 || isempty(droptol))
                droptol = 1E-2;
            end
            
            if(exist('ichol','builtin') > 0)
                opts.type='ict';
                opts.droptol=droptol;
                R=ichol(A,opts);
            else
                R=cholinc(A,droptol);
            end
        end
        
        function [L,U] = luinc_nofill(A)
            % Sparse incomplete LU factorization
            [L,U]=luinc(A,'0');            
        end
        
        function [L,U] = luinc_droptol(A,droptol)
            % Sparse incomplete LU factorization
            if(nargin<2 || isempty(droptol))
                droptol = 1E-2;
            end
            [L,U]=luinc(A,droptol);
        end
        
        function [A,ej,mj,M,iid]=sparse_sparse(jid,A,ej,mj,M)
            A=A(:,jid);
            [iid,~]=find(A);
            iid=unique(iid);
            A=A(iid,:);
            ej=ej(iid);
            if(~isempty(mj))
                mj=mj(jid);
            end
            if(~isempty(M)) % check!!!! 
                M=M(:,iid);
            end
        end               
        
        function [jnew,norm_Aek]=...
                jnew_adaptative(A,norm_Aek,rj,rj2,nnz,jid)
            % Input
            % A   is the coeff. matrix.
            % NORM_AEK is equals to the euclidean norm ||Aek||.
            % RJ  is the j-th residue ej-A*mj            
            % RJ2 is the square of the residue norm.
            % NNZ specifies how many entries will be added to jid.
            % JID is the current sparsity pattern of mj
            
            if(isempty(jid))
                [~,jnew]=find(A(logical(rj),:));
                jnew=ainv.unique(jnew);
            else
                [~,jnew]=find(A(logical(rj),:));
                jnew=setdiff(jnew,sort(jid));
            end
            njnew=length(jnew);
            vnr=zeros(njnew,1);  % the values of the new residual
            rjA=rj'*A(:,jnew);
            for k=1:njnew
                if(norm_Aek(jnew(k))<0)
                    norm_Aek(jnew(k))=norm(A(:,jnew(k)));
                end
                vnr(k)=rj2-(rjA(k)/norm_Aek(jnew(k)))^2;
            end
            nnz=min(njnew,nnz);
            [~,ind]=sort(vnr,'ascend');
            jnew=sort(jnew(ind(1:nnz)),'ascend');
        end                
    end
end