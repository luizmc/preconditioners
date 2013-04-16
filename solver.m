classdef solver
    methods(Static)
        function [niter,relres,tElapsed] = gmres(A,L,R,ainv)
            if(~isempty(ainv) && isempty(L) && isempty(R))
                precond='AINV';
            elseif(~isempty(L) && ~isempty(R))
                precond='LR';
            elseif(~isempty(L) && isempty(R))
                precond='L';
            elseif(isempty(L) && ~isempty(R))
                precond='R';
            else
                precond='NONE';
            end
            
            n = length(A);
            x = zeros(n,1);
            b = A*ones(n,1);
            
            % set parameters
            restart = 30; % the GMRES restarts every RESTART inner iterations
            tol     = 1E-6;
            maxit   = restart*(n^2); % specifies the maximum number of outer iterations
            
            % Syntax of gmres (just a remainder)
            %[x,flag,relres,iter,resvec]=...
            %      gmres(A,b,restart,tol,maxit,M1,M2,x0)
            
            tic
            switch(precond)
                case 'L'
                    [~,flag,relres,~,resvec]=gmres(A,b,restart,tol,maxit,L,[],x);
                case 'R'
                    fA=inline('A*(R\x)','A','R','x');
                    fA=@(x)fA(A,R,x);
                    [x,flag,~,~,resvec]=gmres(fA,b,restart,tol,maxit,[],[],x);
                    x=R\x;
                    relres=norm(b-A*x)/norm(b);
                case 'LR'
                    fA=inline('A*(R\x)','A','R','x');
                    fA=@(x)fA(A,R,x);
                    [x,flag,~,~,resvec]=gmres(fA,b,restart,tol,maxit,L,[],x);
                    x=R\x;
                    relres=norm(b-A*x)/norm(b);
                case 'AINV'
                    [~,flag,relres,~,resvec]=...
                        gmres(A,b,restart,tol,maxit,ainv.apply,[],x);
                case 'NONE'
                    [~,flag,relres,~,resvec]=gmres(A,b,restart,tol,maxit,[],[],x);
            end
            
            if(flag>0)
                switch(flag)
                    case 1
                        error('SOLVER:GMRES','gmres iterated maxit times but did not converge.');
                    case 2
                        error('SOLVER:GMRES','Preconditioner M was ill-conditioned.');
                    case 3
                        error('SOLVER:GMRES','gmres stagnated.');
                end
            end
            
            tElapsed=toc;
            niter = length(resvec) - 1;
        end
    end
end