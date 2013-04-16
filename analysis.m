classdef analysis
    methods(Static)
        function start()
            n         = 100;
            profiling = false;
            plot_fig  = false;
            
            % set all available preconditioner methods
            method = {...                  ids
                @precond.ichol_nofill,...   1 
                @precond.ichol_ict,...      2
                @precond.luinc_nofill,...   3
                @precond.luinc_droptol,...  4
                @precond.spai,...           5
                @precond.spai_s,...         6
                @precond.spai_a,...         7
                @precond.mr_s,...           8
                @precond.mr_a,...           9                
                @precond.bcj,...            10                
                @precond.mr_ap,...          11               
                @precond.fsai,...           12
                };
            
            % set methods to SPD matrices
            method_spd=method([1 2 5:10]);
            
            % set methods to general matrices
            method_general=method([3 4 5:10]);
            
            % numerical experiments with SPD matrices
            instance_name='tridiag';            
            A = analysis.getA(n,@(n)gallery('tridiag',n,-1,2,-1));            
            analysis.precond(method_spd,...
                A,instance_name,profiling,plot_fig)
            
            instance_name='sparse_laplacian 2D';
            A = analysis.getA(n,@(n)sparse_laplacian([n,n],[],[],[]));
            analysis.precond(method_spd,...
                -A,instance_name,profiling,plot_fig)
            
            instance_name='sparse_laplacian 3D';
            A = analysis.getA(n,@(n)sparse_laplacian([n,n,n],[],[],[]));
            analysis.precond(method_spd,...
                -A,instance_name,profiling,plot_fig)            
            
            % numerical experiments with general matrices
            instance_name='5-point Laplacian (butterfly)';
            A = analysis.getA(n,@(n)delsq(numgrid('B',n)));
            analysis.precond(method_general,...
                A,instance_name,profiling,plot_fig)
            
            instance_name='5-point Laplacian (cardioid)';
            A = analysis.getA(n,@(n)delsq(numgrid('H',n)));
            analysis.precond(method_general,....
                A,instance_name,profiling,plot_fig)                        
            
            instance_name='MatrixMarket bcsstk01';
            A=mmread('matrices/bcsstk01.mtx');
            analysis.precond(method_general,....
                A,instance_name,profiling,plot_fig)                        
        end                
        
        function precond(method,A,instance_name,profiling,plot_fig)
            % >> INPUT
            % METHOD is a cell array with function handle to the
            % preconditioners methods.
            % 
            % >> EXAMPLE
            % A=gallery('tridiag',n,-1,2,-1);            
            % method = {...
            %    @precond.ichol_nofill,...    
            %    @precond.ichol_ict,...       
            %    @precond.spai_s,...          
            %    @precond.spai_a,...          
            %    @precond.mr_s,...            
            %    @precond.mr_a,...                            
            %    @precond.bcj,...             
            %    @precond.mr_ap,...           
            %    @precond.fsai...             
            %    };
            % profiling=false;
            % plot_fig =false;
            % precond(method,A,instance_name,profiling,plot_fig)            
                        
            if(nargin<2)
                profiling=false;
                plot_fig=false;
            end                       
                        
            % initializes variables
            nmethod=length(method);
            method_name=cell(nmethod + 1,1);
            
            % set method's name
            method_name{1}='GMRES';
            for i = 2:length(method_name)
                full_name=func2str(method{i-1});
                nickname=upper(strrep(full_name(9:end),'_','-'));
                method_name{i}=nickname;
            end
            
            close all;            
            % solve the original problem using gmres without precond
            try
                [niterGMRES,relres,tElapsed] = solver.gmres(A,[],[],[]);
            catch exception
                warning(['It has not been possible to solve the system ',...
                    'without preconditioning.\n%s'],exception.message);                
            end
            
            % original problem features
            fprintf('\n\nINSTANCE: %s\n',instance_name);            
            analysis.matrix_statistics(A,'A');
            fprintf(['--- GMRES without preconditioning ---\n',...
                'iGMRES : %d\n',...
                'tAX=B  : %f\n',...
                'RELRES : %E\n'   ],...
                niterGMRES,tElapsed,relres);
            
            
            fprintf('--------------------------------------------------------------------\n');
            fprintf('    METHOD    | |I-AM| | tPRECOND |nnzPREC| iGMRES |  tAX=B | RELRES \n')
            if(profiling);profile on;end
            tElapsed=zeros(nmethod,1);
            niter=zeros(nmethod,1);
            for i = 1:nmethod
                [tElapsed(i),niter(i)]=...
                    analysis.call_precond_method(method{i},method_name{i+1},A);
            end
            if(profiling);profile off;profile viewer;end
            
            if(plot_fig)
                figure;
                subplot(2,1,1);
                bar([niterGMRES;niter])
                title('Number of GMRES iterations');
                set(gca,'XTick',1:nmethod+1,'XTickLabel',method_name);
                
                subplot(2,1,2);
                bar(tElapsed);
                title('Elapsed time in seconds');
                set(gca,'XTick',1:nmethod,'XTickLabel',method_name(2:end));
            end
        end
        
        function small(n)
            %A=gallery('poisson',n);
            A=gallery('tridiag',n,-1,2,-1);
            G = precond.fsai(A);
            I = speye(size(A));
            fprintf('|I-G*A*G|=%f\n',precond.norm_frobenius(I-G*A*G'));
            fprintf('|I-G*A*G|=%f\n',precond.norm_frobenius(I-G'*A*G));
        end
        
        function [tABX,niter]=call_precond_method(method,method_name,A)
            P=[];
            L=[];
            R=[];
            ainv=[];
            AM=[];
            nnzA=nnz(A);
            normIAM=nan;
            
            tic;
            % set: P,R,L,M,nnzPRECOND,AM
            if(strncmp(method_name,'ICHOL',5))
                R = method(A);
                L = R';
                tPRECOND=toc;
                nnzPRECOND=nnz(R);
            elseif(strncmp(method_name,'LUINC',5))
                [L,R]=method(A);
                tPRECOND=toc;
                nnzPRECOND=nnz(L)+nnz(R);
            else
                ainv = method(A);
                tPRECOND=toc;
                AM=A*ainv.M;       
                nnzPRECOND=ainv.nnz;
            end
            
            % solves the preconditioned system
            try
                [niter,relres,tABX] = solver.gmres(A,L,R,ainv);
            catch exception                
                error(['It has not been possible to solve the system ',...
                    'with the preconditioner %s.\n%s'],method_name,...
                    exception.message);
            end
                        
            % gets || I-AM ||
            if(~isempty(AM))
                n=length(A);
                I=speye(n);
                normIAM=linalg.norm_frobenius(I-AM);
            end
                        
            fprintf('%13s | %6.2f | %8.2f | %5.2f | %6d | %6.2f | %4.2E \n',...
                method_name,normIAM,tPRECOND,nnzPRECOND/nnzA,niter,tABX,relres);                       
        end
        
        function matrix_statistics(A,matrix_name)
            n = length(A);
            nnzA = nnz(A);
            [i,j,s]=find(A);
            B=sparse(i,j,ones(size(s)),n,n);
            nnzB_row=full(sum(B,2));
            nnzB_col=full(sum(B));
            min_nnz_row = min(nnzB_row);
            min_nnz_col = min(nnzB_col);
            max_nnz_row = max(nnzB_row);
            max_nnz_col = max(nnzB_col);            
            
            fprintf('--- Statistics of matrix %s ---\n',matrix_name);
            fprintf('Size of matrix              = %d\n',n);
            fprintf('Total number of nnz entries = %d\n',nnzA);
            fprintf('Max. nb of nnz per column   = %d\n',max_nnz_col);
            fprintf('Max. nb of nnz per row      = %d\n',max_nnz_row);
            fprintf('Min. nb of nnz per column   = %d\n',min_nnz_col);
            fprintf('Min. nb of nnz per row      = %d\n',min_nnz_row);            
        end
        
        function A = getA(N,fA)
            % Returns the integer N wich minimizes |n^k - length(A)|
            
            % >> INPUT 
            % N  : the expected number of rows (or columns) of A
            % fA : generator function of A            
            
            newN=N;
            A=[];
            while(isempty(A))
                try
                    A=fA(newN);
                catch
                    newN=ceil(0.9*newN);                    
                end
            end
            
            if(length(A)==N)
                A=fA(N);
                return;
            end
            
            minN=1;
            maxN=N;
            while((maxN-minN) > 1)
                newN = floor((minN+maxN)/2);
                if(length(fA(newN)) > N)
                    maxN=newN;
                else
                    minN=newN;
                end
            end            
            if(abs(length(fA(minN)) - N) < abs(length(fA(maxN)) - N))
                newN=minN;
            else
                newN=maxN;
            end
            A = fA(newN);
        end
    end
end