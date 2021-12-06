% INPUT:
% turkish_se_SP500vsMSCI: dataset with 2 colums (column 1 is the input, colums 2 is the output)
% percentage: number indicating the percentage taken from the entire data set to form a new sub set. Number expressed in the interval [0,1]
%
% OUTPUT:
% subSet: dataset with 2 columns as the input but dimension of x% of the input

function [subSet] = subSetCreation(turkish_se_SP500vsMSCI,percentage)
[R,C] = size(turkish_se_SP500vsMSCI);
R_s = round(percentage * R);
randomSubset = randperm(R,R_s);
subSet = zeros(R_s,C);

for i=1:length(randomSubset)
    subSet(i,:) = turkish_se_SP500vsMSCI(randomSubset(i),:);
end
end

