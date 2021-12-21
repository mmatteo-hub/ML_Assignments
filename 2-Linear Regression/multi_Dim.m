function [y,t,w] = multi_Dim(mtcarsdata)
t = mtcarsdata(:,1);
x = mtcarsdata(:,2:end);
w = (pinv(x'*x))*x'*t;
y = x * w;
end