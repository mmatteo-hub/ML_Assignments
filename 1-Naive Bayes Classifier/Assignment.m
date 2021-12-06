%% Initialising the split fraction according to the neural networks standard
% this is the only variable that can be changed by the user
split_frac = .7;

%% Loading the dataset
load('numericdataset.mat');

%% Initilising both training set and data set
rand_vect = randperm(14);
training_set = numericdataset(rand_vect(1:int64(end*split_frac)),:);
test_set = numericdataset(rand_vect(int64(end*split_frac)+1:end),:);

%% Call the function and see the results
[target,classificationNoLp,classificationLp,error_rate] = NaiveBayesClassifier(training_set,test_set);
%% Display the results
if target ~= -1
    disp("Target no Laplace:");
    disp(target(1,:));
    disp("Target with Laplace:");
    disp(target(2,:));
end
disp("Classification no Laplace:");
disp(classificationNoLp);
disp("Classification Laplace:");
disp(classificationLp);
if error_rate ~= -1
    disp("Error Rate no Laplace: "+num2str(error_rate(1,1)));
    disp("Error Rate Laplace: "+num2str(error_rate(2,1)));
end
