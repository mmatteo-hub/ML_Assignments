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

w1 = (sum((x-mean(x)).*(y-mean(y))))./(sum((x-mean(x)).^2));
w0 = mean(y) - w1 * mean(x);

y_c = w0 + w1 .* x;
end

