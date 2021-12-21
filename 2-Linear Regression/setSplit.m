function [subSet,subSet_2] = setSplit(turkish_se_SP500vsMSCI,percentage)
[R,C] = size(turkish_se_SP500vsMSCI);
R_s = round(percentage * R);
randomSubset = randperm(R,R_s);
subSet = zeros(R_s,C);

for i=1:length(randomSubset)
    subSet(i,:) = turkish_se_SP500vsMSCI(randomSubset(i),:);
end
indexes = sort(randomSubset);
subSet_2 = turkish_se_SP500vsMSCI;
for a=1:length(indexes)
    subSet_2(a,:) = [];
end
end

