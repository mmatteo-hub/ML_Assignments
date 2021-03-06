function [target, error_rate] = kNN(X_train,T_train,X_test,k,T_test)
%% Check dims
if nargin() == 5
    t = 1;
elseif nargin() == 4
    t = 0;
else
    disp("Incorrect number of inputs");
    return
end

[n,d] = size(X_train);
[m,d1] = size(X_test);
[n1,c] = size(T_train);
if t
    [m1,c1] = size(T_test);
end

if t
    if n == n1 && m == m1 && d == d1 && c == c1 && c == 1
    else
        disp("Incorrect col num");
        return
    end
else
    if n == n1 && d == d1 && c == 1
    else
        disp("Incorrect row um");
        return
    end
end

if k <= 0
    disp("Invalid k");
elseif k > n
    disp("Invalid k too big");
end

%% Train
target = zeros(m,1);
for i = 1:m
    labels = zeros(k,1);
    distances = zeros(n,1);
    for j = 1:n
        distances(j) = norm(X_train(j,:)- X_test(i,:));
    end
    [~, indexes] = mink(distances,k);
    for h = 1:k
        labels(h) = T_train(indexes(h));
    end
    target(i) = mode(labels);
end
error_rate = 0;
if t
   for l = 1:m
       if target(l) ~= T_test(l)
           error_rate = error_rate + 1/m;
       end
   end
end
