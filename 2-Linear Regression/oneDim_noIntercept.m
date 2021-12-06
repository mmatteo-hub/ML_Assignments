% INPUT:
% turkish_se_SP500vsMSCI: dataset with 2 colums (column 1 is the input, colums 2 is the output)
%
% OUTPUT:
% x: values taken as input
% y: values calculated by y = w * x with w the angular coefficient of the regression


function [x,y] = oneDim_noIntercept(turkish_se_SP500vsMSCI)

w = (turkish_se_SP500vsMSCI(:,1))\(turkish_se_SP500vsMSCI(:,2));
x = turkish_se_SP500vsMSCI(:,1);
y = w * x;
end

