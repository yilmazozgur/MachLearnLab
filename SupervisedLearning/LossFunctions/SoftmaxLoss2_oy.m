function [nll,g,H] = SoftmaxLoss2_oy(w,X,y,k,GPUFlag)
% w(feature*class,1) - weights for last class assumed to be 0
% X(instance,feature)
% y(instance,1)
%
% version of SoftmaxLoss where weights for last class are fixed at 0
%   to avoid overparameterization

[n,p] = size(X);
w = reshape(w,[p k-1]);
w(:,k) = zeros(p,1);


if GPUFlag
    w=gpuArray(w);
end
Z = sum(exp(X*w),2);
nll = -sum((sum(X.*w(:,y).',2) - log(Z)));

if nargout > 1
    g = zeros(p,k-1);
    if GPUFlag
        g=gpuArray(g);
    end
    for c = 1:k-1
        g(:,c) = -sum(X.*repmat((y==c) - exp(X*w(:,c))./Z,[1 p]));
    end
    g = reshape(g,[p*(k-1) 1]);
end

if nargout > 2
    H = zeros(p*(k-1));
    if GPUFlag
        H=gpuArray(H);
    end
    SM = exp(X*w(:,1:k-1))./repmat(Z,[1 k-1]);
    for c1 = 1:k-1
        for c2 = 1:k-1
            D = SM(:,c1).*((c1==c2)-SM(:,c2));
            H((p*(c1-1)+1):p*c1,(p*(c2-1)+1):p*c2) = X'*diag(sparse(D))*X;
        end
    end
end

%%FOR DEBUGGING! Remove Later!
% GPUFlag=false;

if GPUFlag
    nll=gather(nll);
    if nargout > 1
        g=gather(g);
    end
    if nargout > 2
        H=gather(H);
    end
end
    