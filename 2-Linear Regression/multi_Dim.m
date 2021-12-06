function [y,t,w] = multi_Dim(mtcarsdata)
t = mtcarsdata(:,1);
x = mtcarsdata(:,2:end);
X = [ones(length(t),1) x];
Xt = pinv(X);
w = Xt * t;
y = X * w;
end

