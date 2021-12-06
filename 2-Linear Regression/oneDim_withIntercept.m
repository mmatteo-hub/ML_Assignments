% INPUT:
% mtcarsdata: dataset with 4 colums:
%   - colum 1: represents mpg (input for the linear regression)
%   - colum 4: represents weight (output for the linear regression)
%
% OUTPUT:
% x: x values of the data set
% y: values used as output for the linear regression
% y_c: y values calculated as X * b where b is the intercept computed

function [x,y,y_c] = oneDim_withIntercept(mtcarsdata)
x = mtcarsdata(:,4);
y = mtcarsdata(:,1);

X = [ones(length(x),1) x];
b = X\y;

y_c = X * b;
end

