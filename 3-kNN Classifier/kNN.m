function [target, error_rate] = kNN(X_train,T_train,X_test,K,T_test)

if nargin() == 5
    t = 1;
elseif nargin() == 4
    t = 0;
end

[n,d] = size(X_train);
[m,d1] = size(X_test);
[n1,c] = size(T_train);
if t == 1
    [m1,c1] = size(T_test);
end
%% Check dimensions and k
if(d1 ~= d || K <= 0 || K > n)
    error("Inputs not correct!");
end

target = zeros(m,1);

%% Classify the test according to the kNN rule
for i=1:m
    vec = zeros(K,1);
    D = zeros(m,1);
    for j=1:n
        D(i,j) = pdist2(X_train(j,:),X_test(i,:));
    end
    [~,index_min] = mink(D,K);
    for h=1:K
        vec(h,1) = T_train(index_min(h,1),1);
    end
    target(i,1) = mode(vec,1);
end
error_rate = 0;
if t
    for a=1:m
        if target(a,1) ~= T_test(a,1)
            error_rate = error_rate + 1/length(target);
        end
    end
end
end

