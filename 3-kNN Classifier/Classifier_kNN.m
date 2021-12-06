clear;
clc;
close all;
addpath('./mnist');

%% Load data
[X_train, T_train] = loadMNIST(0,0:9);
[X_test, T_test] = loadMNIST(1,0:9);
K = 1;

[target, error_rate] = kNN(X_train(1:600,:),T_train(1:600,:),X_test(1:20,:),K,T_test(1:20,:));

%% Testing the classifier for differnt Ks
K_array = [1,2,3,4,5,10,15,20,30,40,50];

row_tr = 2000;
row_ts = 150;

err_matrix = zeros(10,length(K_array));
for i=1:length(K_array)
    for v=1:10
        if(v == 10)
            [X_t,T_t] = loadMNIST(1,0);
            [target_f, err_matrix(v,i)] = kNN(X_train(1:row_tr,:),T_train(1:row_tr,:),X_t(1:row_ts,:),K_array(i),T_t(1:row_ts,:));
        else
            [X_t,T_t] = loadMNIST(1,v+1);
            [target_f, err_matrix(v,i)] = kNN(X_train(1:row_tr,:),T_train(1:row_tr,:),X_t(1:row_ts,:),K_array(i),T_t(1:row_ts,:));
        end
    end
end

figure;
for i=1:length(err_matrix(1,:))
    hold on;
    subplot(1,length(K_array),i);
    bar(err_matrix(:,i));
end

accuracy = 1-err_matrix;
figure;
for i=1:length(accuracy(:,1))
    hold on;
    %figure(i)
    plot(K_array, accuracy(i,:),'LineWidth',2);
    axis([min(K_array), max(K_array),.5,1]);
    title('Accuracy for kNN classifer')
    xlabel('K classes');
    ylabel('Accuracy');
    legend('Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9');
    grid
end