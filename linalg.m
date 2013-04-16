classdef linalg
    methods(Static)
        function normA=norm_frobenius(A)
            if(issparse(A))
                [~,~,s]=find(A);
                normA=sum(s.*s);
            else
                normA=sum(sum(A.*A));
            end
            normA=sqrt(normA);
        end
        
        function AB=innerprod(A,B)
            % Calculates the inner product on the space of matrices
            % associated to the Frobenius norm.
            %
            % Output:
            % AB :: <A,B>=tr(A'*B);
            %
            [n,~]=size(A);
            AB=0;
            for i = 1:n
                AB=AB+sum(A(:,i).*B(:,i));
            end
        end
    end
end