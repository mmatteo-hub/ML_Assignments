clear;
clc;
close all;
addpath('./mnist');

%% Load data
[X_train, T_train] = loadMNIST(0,1:9);
[X_test, T_test] = loadMNIST(1,1:9);
K = 1;

[target, error_rate] = kNN(X_train(1:600,:),T_train(1:600,:),X_test(1:20,:),K,T_test(1:20,:));

%% Testing the classifier for differnt Ks
K_array = [1,2,3,4,5,10,15,20,30,40,50];

err_matrix = zeros(10,length(K_array));
for i=1:length(K_array)
    for v=1:10
        [X_t,T_t] = loadMNIST(1,v);
        [target, err_matrix(v,i)] = kNN(X_train(1:1000,:),T_train(1:1000,:),X_t(1:100,:),K_array(i),T_t(1:100,:));
    end
    
    %hold on;
    %bar(err_matrix(:,i));
end

accuracy = 1-err_matrix;
figure;
for i=1:10
    hold on;
    %figure(i)
    plot(K_array, accuracy(i,:),'LineWidth',2);
    axis([min(K_array), max(K_array),.5,1]);
    title('kNN classifer: Accuracy with 3000 rows of training and 200 rows of test.')
    xlabel('K classes');
    ylabel('Accuracy');
    legend('Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9');
    grid
end