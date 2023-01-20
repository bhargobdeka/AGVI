%% Online Inference of a 3x3 Q Matrix using Adaptive Kalman Filter (AKF)%%
%% ----------Code is implemented from Dunik et al. (2017) -----------------------
% Created by Bhargob Deka and James-A. Goulet, 2022
%%
clear;clc
% rand_seed=4;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
N   = 1e3;  % number of simulation steps
n_x = 3; % no. of hidden states
sV  = 1e-02; % observation error stdv.
%% Chooose A and true variance terms
s11 = 1; s22 = 2; s33 = 4; % variance terms
s12 = -0.5; s13 = -0.3; s23 = 0.95; % covariance terms

%% A matrix
A = diag(1*(ones(1,n_x)));
%% Q matrix
corr       = [s11 s12 s13;... 
              s12 s22 s23;...
              s13 s23 s33];
Q_true     =  eye(3)*corr;
%% System parameters
sys.R = diag(sV^2*ones(1,n_x));
sys.Q = Q_true;
sys.F = A;
sys.H = diag(ones(1,n_x));

% initial estimate
param.xp = zeros(n_x,1); % initial state estimate
param.lags = 1; % # of time lags
param.K = diag(0.5*ones(1,n_x)); % stable linear filter gain

%% ICM PARAMETERS 
%--------------------------------------------------
ICMpar.xp = param.xp; % initial state estimate
ICMpar.lags = param.lags; % time lag of autocovariance function
ICMpar.K = param.K; % stable linear filter gain

%% CMM PARAMETERS
%--------------------------------------------------
% initial state estimate
CMMpar.xp = param.xp;
CMMpar.Pp = 1*eye(n_x);

% initial estimates of Q and R
CMMpar.Q = eye(n_x);
CMMpar.R = eye(n_x);

% initial time instant for matrices estimation
CMMpar.erq = floor(N/4);%N/2
no_of_datasets = 5;
%% Looping for each synthetic dataset to estimate Q matrix
for j = 1:no_of_datasets
    filename = sprintf('Datasets_n_3/Dataset%d.mat',j);
    MC = 1e0;
    cntMethods = 2;
    est = cell(cntMethods,MC);
    for imc = 1:MC
        dat = load(filename);
        z   = dat.YT;
    
      for i = 1:cntMethods
        switch i
          case 1
            est{i,imc} = ICM(sys,z,ICMpar);
          case 2
            est{i,imc} = CMM(sys,z,CMMpar);
        end
      end
    end
    %% Get the estimated Q matrices
    methods = {'ICM','CMM'};%'ICM',
    for i=1:cntMethods
      fprintf('Method %s\n',methods{i});
      Q_est = est{i,1}.Q;
      save(['Q_AKF_results/Q_' methods{i} '_Dataset' num2str(j) '.mat'],'Q_est')
    end
end