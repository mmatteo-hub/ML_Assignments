function [target, error_rate] = kNN(X_train,T_train,X_test,K,T_test)
%% Checking dimensions
if nargin() == 5
    test_target_given = 1;
elseif nargin() == 4
    test_target_given = 0;
else
    disp("Incorrect number of inputs");
    return
end

%% initialization
[n,d] = size(X_train);
[m,d1] = size(X_test);
[n1,c] = size(T_train);
if test_target_given
    [m1,c1] = size(T_test);
end
    
if test_target_given
    if n == n1 && m == m1 && d == d1 && c == c1 && c == 1
    else
        disp("Error");
        return
    end
else
    if n == n1 && d == d1 && c == 1
    else
        disp("Error");
        return
    end
end

if K <= 0
    disp("k negative");
elseif K > n
    disp("k too big");
end

%% Building the classifier
target = zeros(m,1);
for i = 1:m
    fdist = zeros(K,1);
    errors = zeros(n,1);
    for h = 1:n
        errors(h,1) = norm(X_train(h,:)- X_test(i,:));
    end
    [~, index] = mink(errors,K);
    for k = 1:K
        fdist(k,1) = T_train(index(k,1), 1);
    end
    target(i,1) = mode(fdist,1);
end
error_rate = 0;
if test_target_given
   for l = 1:m
       if T_test(l,1) ~= target(l,1)
           error_rate = error_rate + 1 / length(target);
       end
   end
end
end

