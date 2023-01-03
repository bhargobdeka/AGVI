clear;clc
% syms d1 d2 d3 c1 c2 c3 real
% A = [ d1 c1 c2;...
%       0  d2 c3;...
%       0  0  d3];
% Sigma_W = A'*A;
%% Parameters
% A matrix
A_LL = 0.8;
A_T  = blkdiag(1,1,1);
% Q matrix
T=1000;                 %Time-serie length
n_x        = 3;
n_w        = n_x;
n          = n_x*2;
n_w2hat    = n_x*(n_x+1)/2;
y          = zeros(T,1);        %  Initialization of the vector of observations
sV         = 1e-02;
R_T        = sV^2.*eye(n_x);
% [Q,corr_Q] = make_Q(3,0.4,0.8);
% corr       = [1 0.1 0.1;...
%               0.1 1 0.1;...
%               0.1 0.1 1];
s12 = 0.62; s13 = 0.8; s23 = -0.4;
corr       = [1 s12 s13;... s12 = 0.95, s13 = 0.99, s23 = 0.95
              s12 2 s23;...
              s13 s23 4];
Q          =  eye(3)*corr;
eig(Q)
sW         = diag(Q)';
sW_cov     = [Q(1,2:3)';Q(2,3)];
%% Data
YT          = zeros(n_x,T);
x_true      = zeros(n_x,T);
x_true(:,1) = [0;0;0];
w           = chol(Q)'*randn(n_x,T);
v           = chol(R_T)'*randn(n_x,T);
C_LL        = 1;
C_T         = blkdiag(C_LL,C_LL,C_LL);
for t = 2:T
    x_true(:,t)  = A_T*x_true(:,t-1) + w(:,t);
    YT(:,t)      = C_T*x_true(:,t)   + v(:,t);
end
% figure;
% plot(YT(1,:))
%% Detecting and filling outliers (more than 3 std from mean)
% YT = filloutliers(YT,'center','mean','ThresholdFactor', 3); 

%% Initial value in Cholesky-space E[L], var[L]
n           = 3;
total       = n*(n+1)/2;
mL          = [1.5*ones(n,1);0.1;0.2;0.3];
SL          = [0.15*ones(n,1);0.5*ones(total-n,1)];
ind         = [1 4 2 5 6 3];
mL = mL(ind);SL = diag(SL(ind));
%% State Estimation
% initialization of L
E_Lw        = zeros(total,T);
P_Lw        = zeros(total,total,T);
EX          = zeros(n_w+n_w2hat,T);
E_Lw(:,1)   = mL;                 
P_Lw(:,:,1) = SL;
EX(:,1)     = [zeros(1,n_x) mL']';    %[E[\bm{X}]  E[\bm{W2hat}]]                                 
PX(:,:,1)   = diag([1*ones(1,n_x), diag(SL)']);    %[P[\bm{X}]  P[\bm{W2hat}]] 
eig_Sp      = zeros(n,T);
index_eig   = zeros(1,T);
ind_covij   = multiagvi.index_covij(n);
n_w2        = n_x*(n_x+1)/2;
%% Indices
% Creating the cells that will hold the covariance terms
% cov(w_iw_j,w_kw_l) and the indices for each row 
ind_wijkl   = multiagvi.indcov_wijkl(n_x);
%% Getting the mean indices
ind_mu           = multiagvi.ind_mean(n_x,ind_wijkl);
%% Getting the covariance indices from the \Sigma_W matrix
ind_cov          = multiagvi.ind_covariance(n_w2,n_w,ind_wijkl);
%% Computing the indices for the covariances for prior \Sigma_W^2:
ind_cov_prior    = multiagvi.indcov_priorWp(ind_wijkl,n_w2,n_x);   
start = tic;
for t=2:T
    Ep      = [A_T*EX(1:n_x,t-1); zeros(n_x,1)];           % mu_t|t-1
    mL      = E_Lw(:,t-1);
    SL      = P_Lw(:,:,t-1);
    %% Matrices for mL and SL
    mL_mat = triu(ones(n));
    mL_mat(logical(mL_mat)) = mL;
    SL_mat = triu(ones(n));
    SL_mat(logical(SL_mat)) = diag(SL);
%     m      = logical(triu(mL_mat));
    %% Initialization for Product L^T*L
    [E_P,P_P,Cov_L_P] = multiagvi.LtoP(n,mL_mat,SL_mat,SL,ind_covij);
    % Covariance matrix construction for Prediction Step : Sigma_t|t-1
    Q_W  = triu(ones(n));
    Q_W(logical(Q_W)) = E_P;
    Q_W = triu(Q_W)+triu(Q_W,1)';
    C   = [eye(n_x) zeros(n_x)];
    %%  1st update step:
    [EX1,PX1] = multiagvi.KFPredict(A_T,C,Q_W,R_T,YT(:,t),PX(1:n_x,1:n_x,t-1),Ep);
    PX1 = (PX1+PX1')/2;
    EX(1:n_w,t)       = EX1(1:n_w);    % n = n_x*2 i.e., no of x + no of w
    PX(1:n_w,1:n_w,t) = PX1(1:n_w,1:n_w);
    %% Collecting W|y
    EX_wy   = EX1(end-n_x+1:end,1);
    PX_wy   = PX1(end-n_x+1:end,end-n_x+1:end,1);
    m       = triu(true(size(PX_wy)),1);  % Finding the upper triangular elements
    cwiwjy  = PX_wy(m)';                  % covariance elements between W's after update
    P_wy    = diag(PX_wy); % variance of updated W's
    
    %%% 2nd Update step: 
    %% Computing W^2|y
    [m_wii_y,s_wii_y,m_wiwj_y, s_wiwj_y] = multiagvi.meanvar_w2pos(EX_wy,PX_wy,cwiwjy, P_wy, n_w2hat,n_x);

    %% Computing the covariance matrix \Sigma_W^2|y:
    cov_wijkl = multiagvi.cov_w2pos(PX_wy,EX_wy,n_w2,ind_cov,ind_mu);
    
    %% Adding the variances and covariances of W^p to form \Sigma_W^p
    PX_wpy    = multiagvi.var_wpy(cov_wijkl,s_wii_y,s_wiwj_y);
    
    %% Creating E[W^p]
    EX_wpy        = [m_wii_y' m_wiwj_y];
    
    %% Computing prior mean and covariance matrix of Wp
    m_wsqhat    = E_P;
    s_wsqhat    = P_P;
    PX_wp       = multiagvi.covwp(P_P,m_wsqhat,n_x,n_w2,n_w);
    
    %% Computing the prior covariance matrix \Sigma_W^p
    cov_prior_wijkl = multiagvi.priorcov_wijkl(m_wsqhat,ind_cov_prior,n_w2);
    %% Adding the variances and covariances for Prior \Sigma_W^p
    cov_wp              = cell2mat(reshape(cov_prior_wijkl,size(cov_prior_wijkl,2),1));
    s_wp                = zeros(size(PX_wp,1));
    s_wp(1:end-1,2:end) = cov_wp;
    PX_wp               = PX_wp + s_wp;
    PX_wp               = triu(PX_wp)+triu(PX_wp,1)'; % adding the lower triangular matrix

    %% Creating Prior E[W^p]
    E_wp        = m_wsqhat;
    %% Smoother Equations
    [ES,PS] = multiagvi.agviSmoother(E_wp,s_wsqhat,PX_wp,EX_wpy,PX_wpy,E_P,P_P);
    EX(end-n_w2+1:end,t)                 = ES;
    PX(end-n_w2+1:end,end-n_w2+1:end,t)  = PS;
    
    E_Pw_y = EX(end-n_w2+1:end,t);
    P_Pw_y  = PX(end-n_w2+1:end,end-n_w2+1:end,t);
    P_Pw_y  = (P_Pw_y+P_Pw_y')/2;
    
    %% Converting from Pw to Lw
    [EL_pos,PL_pos] = multiagvi.PtoL(Cov_L_P,E_P,P_P,E_Lw(:,t-1),P_Lw(:,:,t-1),E_Pw_y,P_Pw_y);
    E_Lw(:,t)   = EL_pos;
    P_Lw(:,:,t) = PL_pos;
end
runtime = toc(start)
%% Plotting
disp_plot = 1;
if disp_plot==1
%  Plotting Variances
    E_W = zeros(n_w*2,length(EX));
    V_W = zeros(n_w*2,length(EX));
    for t = 1:length(EX)
        mL     = E_Lw(:,t);
        SL     = P_Lw(:,:,t);
        %% Matrices for mL and SL
        mL_mat = triu(ones(n));
        mL_mat(logical(mL_mat)) = mL;
        SL_mat = triu(ones(n));
        SL_mat(logical(SL_mat)) = diag(SL);
    %     m      = logical(triu(mL_mat));
        %% Initialization for Product L^T*L
        [E_W(:,t),P_P(:,:,t)] = multiagvi.LtoP(n,mL_mat,SL_mat,SL,ind_covij);
        V_W(:,t)              = diag(P_P(:,:,t));
    end
    t  = 1:length(EX);
    figure;
    for i=1:n_x
        subplot(n_x,1,i)
        
        xw = E_W(i,t);
        sX = sqrt(V_W(i,t));
        plot(t,repmat(sW(i),[1,length(EX)]),'-.r','Linewidth',1)
        hold on;
        patch([t,fliplr(t)],[xw+sX,fliplr(xw-sX)],'g','FaceAlpha',0.2,'EdgeColor','none')
        hold on;
        plot(t,xw,'k')
        hold off
        xlabel('$t$','Interpreter','latex')
        ylabel(['$\sigma^{\mathtt{AR}}' num2str(i) '$'],'Interpreter','latex')
        ylim([0,5])
        title('variance')
    end
    % Plotting Covariances
    figure;
    for i=1:n_w2-n_w
        subplot(n_x,1,i)
        xw = E_W(n_w+i,t);
        sX = sqrt(V_W(n_w+i,t));
        plot(t,repmat(sW_cov(i,1),[1,length(EX)]),'-.r','Linewidth',1)
        hold on;
        patch([t,fliplr(t)],[xw+sX,fliplr(xw-sX)],'g','FaceAlpha',0.2,'EdgeColor','none')
        hold on;
        plot(t,xw,'k')
        hold off
        xlabel('$t$','Interpreter','latex')
        ylabel(['$\sigma^{\mathtt{AR}}' num2str(i) '$'],'Interpreter','latex')
        ylim([-2,2])
        title('covariance')
    end
end
