clear;
clc;
close all;
addpath('./mnist');

%% Task 2: Autoencoder

[X, T] = loadMNIST(0,1:4);

% take 10% of the total
rows = randperm(12700,floor(0.10*12700));
X_sub = X(rows,:);
T_sub = T(rows,:);

nh = 2;
myAutoencoder = trainAutoencoder(X_sub',nh);
[Xenc, Tenc] = loadMNIST(0,1:4);

rows = randperm(12700,floor(0.10*12700));
X_sub_enc = Xenc(rows,:);
T_sub_enc = Tenc(rows,:);

myEncodedData = encode(myAutoencoder, X_sub_enc');

%% Plot
addpath('./plot');
figure
plotcl(myEncodedData',T_sub_enc);
%text(T_sub_enc(:,1),T_sub_enc(:,1),'');

