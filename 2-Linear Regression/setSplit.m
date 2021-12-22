function [subSet,subSet_2] = setSplit(dataset,percentage)
[R,~] = size(dataset);
R_s = ceil(percentage * R);
randomSubset = randperm(R);

subSet = dataset(randomSubset(1:R_s),:);

subSet_2 = dataset(randomSubset(R_s+1:end),:);

end

