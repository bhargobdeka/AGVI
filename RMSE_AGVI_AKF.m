%% RMSE Comparison for AGVI vs ICM, CMM, and SWVBAKF
% Created by Bhargob Deka and James-A. Goulet, 2022 %
%%
clear;clc
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
format short
True    = [1;-0.5;-0.3;2;0.95;4];
folder  = {'AGVI','AKF','AKF','SWVBAKF'};
methods = {'AGVI','ICM','CMM','SWVBAKF'};
M       = length(methods);
Q       = cell(M,1);
n_x     = 3;                    % no. of hidden states
n_w     = n_x;                  % no. of process error terms
n_w2hat = n_x*(n_x+1)/2;        % total variance and covariance terms   
Cov     = zeros(n_w2hat,M);
RMSE    = zeros(n_w2hat,M);
for i = 1:length(methods)
    Q{i} = load(['Q_' folder{i} '_results/Q_' methods{i} '_Dataset1.mat']);
    if strcmp(methods{i},'AGVI')
        Q_est = Q{i,1}.Q_mat;
    elseif strcmp(methods{i},'ICM')
        Q_est = Q{i,1}.Q_est;
    elseif strcmp(methods{i},'CMM')
        Q_est = Q{i,1}.Q_est;
    elseif strcmp(methods{i},'SWVBAKF')
        Q_est = Q{i,1}.new_Qapriv;
    end
    ind = tril(true(size(Q_est)));
    Cov(:,i)  = Q_est(ind);
    RMSE(:,i) = sqrt(mean((True - Cov(:,i)).^2));
end





