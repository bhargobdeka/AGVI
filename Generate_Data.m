%% Online Inference of a 3x3 Q Matrix %%
% Created by Bhargob Deka and James-A. Goulet, 2022
%%
clear;clc
% rand_seed=4;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
%% Chooose A and true variance terms
A = 1; % A value for each time series
s11 = 1; s22 = 2; s33 = 4; % variance terms
s12 = -0.5; s13 = -0.3; s23 = 0.95; % covariance terms
%% Parameters
T          = 1000;                 % Time-serie length
n_x        = 3;                    % no. of hidden states
n_w        = n_x;                  % no. of process error terms

n_w2hat    = n_x*(n_x+1)/2;        % total variance and covariance terms   
y          = zeros(T,1);           % initialization of the vector of observations
sV         = 1e-02;                % observation error variance
R_T        = sV^2.*eye(n_x);
%% A matrix
A_G  = blkdiag(A,A,A);     % Global A matrix
%% Q matrix
corr       = [s11 s12 s13;... 
              s12 s22 s23;...
              s13 s23 s33];
Q          =  eye(3)*corr;
%% Check if matrix is psd or not
Eg = eig(Q);
if any(Eg(Eg<0))
    error('Q matrix is non-PSD!')
end
sW         = diag(Q)'; % variance terms
sW_cov     = [Q(1,2:3)';Q(2,3)]; %covariance terms
%% Data
N = 5; % no. of simulated datasets
for i =1:N
    YT          = zeros(n_x,T);
    x_true      = zeros(n_x,T);
    x_true(:,1) = [0;0;0];
    w           = chol(Q)'*randn(n_x,T);
    v           = chol(R_T)'*randn(n_x,T);
    C_LL        = 1;
    C_T         = blkdiag(C_LL,C_LL,C_LL);
    for t = 2:T
        x_true(:,t)  = A_G*x_true(:,t-1) + w(:,t);
        YT(:,t)      = C_T*x_true(:,t)   + v(:,t);
    end
    % Save simulated datasets
    filename = sprintf('Datasets_n_3/Dataset%d.mat',i);
    save(filename,'YT')
end