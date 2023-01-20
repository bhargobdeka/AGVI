%% Online Inference of a 3x3 Q Matrix using Slide Window Variational Adaptive Kalman Filter (VBAKF)%%
%% ----------Code is implemented by Huang et al. (2019) -----------------------
% Created by Bhargob Deka and James-A. Goulet, 2022
%%
clear;
close all;clc;
format long;
%%% New Parameters
ts=1000;
%%%%%Model parameters
nxp= 100;
n_x = 3;
nz = 3;
T  = 1; 
sV = 1e-02;
R  = sV^2*eye(n_x);
%% Choose Variance and Covariance terms
s11 = 1; s22 = 2; s33 = 4; % variance terms
s12 = -0.5; s13 = -0.3; s23 = 0.95; % covariance terms
%% Q matrix
corr       = [s11 s12 s13;... 
              s12 s22 s23;...
              s13 s23 s33];
Q_true     =  eye(n_x)*corr;
Q          = Q_true;
A          = diag(1*(ones(1,n_x)));

Q1         = Q;
R1         = R;
F          = A;
H          = eye(n_x);

%% Selections of filtering parameters
N=50;                 
tao_P=5;  
tao_R=5; 
rou=1-exp(-4);
beta=10;
Q_s=10;
alfa=1;

%% Loading data
no_of_datasets = 5;
for j = 1:no_of_datasets
filename = sprintf('Datasets_n_3/Dataset%d.mat',j);
load(filename);
    %% Monte-Carlo Runs
    runtime_s = tic;
    for expt=1:nxp
        
        fprintf('MC Run in Process=%d\n',expt); 
        
        %%%%%Set the system initial value%%%%%%%%%%%
        x=10*ones(n_x,1);                     %%%True initial state value 
        P=diag(10*ones(n_x,1));              %%%Initial estimation error covariance matrix 
        Skk=utchol(P);                         %%%Square-root of initial estimation error covariance matrix
        
        %%%%Nominal measurement noise covariance matrix (R)
        R0=beta*eye(nz);
        
        %%%%Initial state estimate of standard KF    (Kalman filter)
        xf=x+Skk*randn(n_x,1);               
        Pf=P;
        
        %%%%Initial state estimate of Kalman filter with true noise covariance matrices    (Optimal Kalman filter)
        xtf=xf;
        Ptf=Pf;
    
        %% Initial state estimate of adaptive methods
        xiv=xf;
        Piv=Pf;
    
        %%%%
        xA=[];
        ZA=[];
        
        for i=1          %:length(alfa)
            
            %%%%Nominal state noise covariance matrix (Q)
            Q0=alfa(i)*eye(n_x);
            
            %%%%Initial state estimate of VBAKF-PR
            xapriv=xiv;
            Papriv=Piv;
            uapriv=tao_R;
            Uapriv=tao_R*R0;
            
            %%%%Initial state estimate of proposed SWVAKF
            new_xapriv=xiv;
            new_Papriv=Piv;
            new_xapriv_A=[];
            new_Papriv_A=[];
            new_yapriv=0;
            new_Yapriv=zeros(n_x);
            new_uapriv=0;
            new_Uapriv=zeros(nz);
            new_Qapriv=Q0;
            new_Rapriv=R0;
            new_Lapriv=5;
            zA=[];
    
            for t=1:ts
                
                %%%%True noise covariance matrices
                Q=Q1;
                R=R1;
                
                %%%%Extract data
    %             x = x_true(:,t);
                z = YT(:,t);            
                
                %%%%Save measurement data
                if t<=(new_Lapriv+1)
                    zA=[zA z];
                else
                    zA=[zA(:,2:end) z];
                end
                
                %%%%Run VBAKF-PR and proposed SWVAKF
    %             [xapriv,Papriv,uapriv,Uapriv,Ppapriv,Rapriv,Qapriv]=aprivbkf(xapriv,Papriv,uapriv,Uapriv,F,H,z,Q0,R,N,tao_P,rou);
                
                [new_xapriv,new_Papriv,new_xapriv_A,new_Papriv_A,new_yapriv,new_Yapriv,new_uapriv,new_Uapriv,new_Qapriv,R,new_Ppapriv]=...
                new_aprivbkf(new_xapriv,new_Papriv,new_xapriv_A,new_Papriv_A,new_yapriv,new_Yapriv,new_uapriv,new_Uapriv,F,H,zA,new_Qapriv,R,rou,new_Lapriv,t);
    
                
                
            end
            
        end
        
    end
    runtime_e(j) = toc(runtime_s)
    save(['Q_SWVBAKF_results/Q_SWVBAKF_Dataset' num2str(j) '.mat'],'new_Qapriv')
end

