%% RMSE Comparison for AGVI vs ICM, CMM, and SWVBAKF
% Created by Bhargob Deka and James-A. Goulet, 2022 %
%%
clear;clc
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
format short
True_var    = [1;-0.5;-0.3;2;0.95;4];%s11, s12, s13, s22, s23, s33
folder  = {'AGVI','AKF','AKF','SWVBAKF'};
methods = {'AGVI','ICM','CMM','SWVBAKF'};
M       = length(methods);
Q       = cell(M,1);
n_x     = 3;                    % no. of hidden states
n_w     = n_x;                  % no. of process error terms
n_w2hat = n_x*(n_x+1)/2;        % total variance and covariance terms   
Cov     = zeros(n_w2hat,M);
% RMSE    = zeros(n_w2hat,M);
N = 5;
for j = 1:N % for each synthetic datasets
    for i = 1:length(methods)
        Q{i} = load(['Q_' folder{i} '_results/Q_' methods{i} '_Dataset' num2str(j) '.mat']);
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
        
    end
    CovT = Cov'; % one row represent all variances for one method
    % Cov for each method
    CovT_AGVI(j,:)    = CovT(1,:);
    CovT_ICM(j,:)     = CovT(2,:);
    CovT_CMM(j,:)     = CovT(3,:);
    CovT_SWVBAKF(j,:) = CovT(4,:);
end
for k = 1:length(True_var)
RMSE_AGVI(1,k)      = sqrt(mean((True_var(k) - CovT_AGVI(:,k)).^2));
RMSE_ICM(1,k)       = sqrt(mean((True_var(k) - CovT_ICM(:,k)).^2));
RMSE_CMM(1,k)       = sqrt(mean((True_var(k) - CovT_CMM(:,k)).^2));
RMSE_SWVBAKF(1,k)   = sqrt(mean((True_var(k) - CovT_SWVBAKF(:,k)).^2));
end
RMSE = [RMSE_AGVI;RMSE_ICM;RMSE_CMM;RMSE_SWVBAKF]





