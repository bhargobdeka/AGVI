clear;clc
%% Cholesky matrix
% [ d1(1) d2(2) d3(3) c1(4) c2(5) c3(6)]
% syms d1 d2 d3 d4 d5 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 real
% A = [ d1 c1 c2 c3 c4;...
%       0  d2 c5 c6 c7;...
%       0  0  d3 c8 c9;...
%       0  0  0  d4 c10;...
%       0  0  0  0  d5];
% Sigma_W = A'*A;
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
% Sigma_W =
% [ d1^2,         c1*d1,                 c2*d1,                          c3*d1,                             c4*d1]
% [ c1*d1,   c1^2 + d2^2,         c1*c2 + c5*d2,                  c1*c3 + c6*d2,                     c1*c4 + c7*d2]
% [ c2*d1, c1*c2 + c5*d2,    c2^2 + c5^2 + d3^2,          c2*c3 + c5*c6 + c8*d3,             c2*c4 + c5*c7 + c9*d3]
% [ c3*d1, c1*c3 + c6*d2, c2*c3 + c5*c6 + c8*d3,      c3^2 + c6^2 + c8^2 + d4^2,    c3*c4 + c6*c7 + c8*c9 + c10*d4]
% [ c4*d1, c1*c4 + c7*d2, c2*c4 + c5*c7 + c9*d3, c3*c4 + c6*c7 + c8*c9 + c10*d4, c4^2 + c7^2 + c9^2 + c10^2 + d5^2]
%% Parameters
% A matrix
%% A matrix
A_LL = 1;

%% Q matrix
T=1000;                 %Time-serie length
N=50;
n_x        = 5;
n_w        = n_x;
n          = n_x*2;
n_w2hat    = n_x*(n_x+1)/2;
y          = zeros(T,1);        %  Initialization of the vector of observations
sV         = 1e-02;
R_T        = sV^2.*eye(n_x);
A_T        = diag(1*ones(1,n_x));
% [Q,corr_Q] = make_Q(3,0.4,0.8);
% corr       = [1 0.1 0.1;...
%               0.1 1 0.1;...
%               0.1 0.1 1];
% sig = 0.6*ones(1,10);
% sig        = [0.05 0.22 0.37 0.42 0.59 0.7 0.75 0.83 0.8 0.95];
sig        = [-0.3 -0.2 -0.1 0.25 0.35 0.4 0.45 0.5 0.55 0.6];
corr       = [1 sig(1) sig(2) sig(3) sig(4);...
              sig(1) 3 sig(5) sig(6) sig(7);...
              sig(2) sig(5) 4 sig(8) sig(9);...
              sig(3) sig(6) sig(8) 0.8 sig(10);...
              sig(4) sig(7) sig(9) sig(10) 2];
Q          =  eye(5)*corr;
% Q = [   0.0866    0.0007    0.5114    0.1898    0.2109
%         0.0007    3.0155    0.5374   -0.6165    0.7846
%         0.5114    0.5374    3.5209    1.1580    1.5473
%         0.1898   -0.6165    1.1580    0.6075    0.3616
%         0.2109    0.7846    1.5473    0.3616    0.7945];
QT     = Q.';
m1     = tril(true(size(QT)),-1);
sW     = diag(QT)';
sW_cov = Q(m1);
eig_Q      = eig(Q);
% sW         = diag(Q)';
% sW_cov     = [Q(1,2:5)';Q(2,3:5)';Q(3,4:5)';Q(4,5)];
% Q     = diag([1 3 4 0.8]);
% eig_Q = eig(Q);

%% Data
YT          = zeros(n_x,T);
x_true      = zeros(n_x,T);
x_true(:,1) = zeros(n_x,1);
w           = chol(Q)'*randn(n_x,T);
v           = chol(R_T)'*randn(n_x,T);
C_LL        = 1;
C_T         = diag((C_LL.*ones(1,n_x)));
% for t = 2:T
%     x_true(:,t)  = A_T*x_true(:,t-1) + w(:,t);
%     YT(:,t)      = C_T*x_true(:,t)   + v(:,t);
% end
for p = 1:5
dat = load(['AGVI_M_CS1_Data0' num2str(p) '.mat']);
YT  = dat.YT;
% ind = [1 10 20 40:50 120];
% YT(:,ind)  = nan;
% figure;
% plot(YT(1,:))
%% Detecting and filling outliers (more than 3 std from mean)
% YT = filloutliers(YT,'center','mean','ThresholdFactor', 3); 

%% Initial value in Cholesky-space E[L], var[L]
% n           = n_x;
total       = n_x*(n_x+1)/2;
mL_old      = [2*ones(n_x,1);0.8*ones(total-n_x,1)]; % cannot start at zero have to fix
SL_old      = diag([0.5*ones(n_x,1);0.5*ones(total-n_x,1)]);
% [mL, SL]    = multiagvi.convertstructure(mL_old,SL_old,n);
%% State Estimation
% initialization of L
E_Lw        = zeros(total,T);
P_Lw        = zeros(total,total,T);
EX          = zeros(n_w+n_w2hat,T);
E_Lw(:,1)   = mL_old;                 
P_Lw(:,:,1) = SL_old;
EX(:,1)     = [zeros(1,n_x) mL_old']';    %[E[\bm{X}]  E[\bm{W2hat}]]                                 
PX(:,:,1)   = diag([1*ones(1,n_x), diag(SL_old)']);    %[P[\bm{X}]  P[\bm{W2hat}]]   % high value of variance is important
% eig_Sp      = zeros(n,T);
% index_eig   = zeros(1,T);
ind_covij   = multiagvi.index_covij(n_x);
NIS_AGVI    = zeros(1,T);
%% Indices
% Creating the cells that will hold the covariance terms
% cov(w_iw_j,w_kw_l) and the indices for each row
n_w2             = n_x*(n_x+1)/2;
ind_wijkl        = multiagvi.indcov_wijkl(n_x);
%% Getting the mean indices
ind_mu           = multiagvi.ind_mean(n_x,ind_wijkl);
%% Getting the covariance indices from the \Sigma_W matrix
ind_cov          = multiagvi.ind_covariance(n_w2,n_w,ind_wijkl);
%% Computing the indices for the covariances for prior \Sigma_W^2:
ind_cov_prior = multiagvi.indcov_priorWp(ind_wijkl,n_w2,n_x);
start = tic;
for t=2:T
%     if t==101
%         stop=1;
%     end
    Ep      = [A_T*EX(1:n_x,t-1); zeros(n_x,1)];           % mu_t|t-1
    mL_old  = E_Lw(:,t-1); %old
    SL_old  = P_Lw(:,:,t-1);%old
    [mL, SL]= multiagvi.convertstructure(mL_old,SL_old,n_x); %converted to new structure
    %% Matrices for mL and SL
    mL_mat  = triu(ones(n_x));
    mL_mat(logical(mL_mat)) = mL;
    SL_mat  = triu(ones(n_x));
    SL_mat(logical(SL_mat)) = diag(SL);
%     SL_mat = (SL_mat+SL_mat')/2;
%     m      = logical(triu(mL_mat));
    %% Initialization for Product L^T*L
    [E_P,P_P,Cov_L_P] = multiagvi.LtoP(n_x,mL_mat,SL_mat,SL,ind_covij);
    P_P = (P_P+P_P')/2;
    %% check
    [E_P_old,P_P_old,Cov_L_P_old] = multiagvi.convertback(E_P,P_P,Cov_L_P,n_x);
     P_P_old = (P_P_old+P_P_old')/2;
    % Covariance matrix construction for Prediction Step : Sigma_t|t-1
    Q_W  = triu(ones(n_x));
    Q_W(logical(Q_W)) = E_P;
    Q_W = triu(Q_W)+triu(Q_W,1)';
    Q_W  = (Q_W + Q_W)'/2;    
    C   = [eye(n_x) zeros(n_x)];
    %%  1st update step:
    [EX1,PX1,NIS] = multiagvi.KFPredict(A_T,C,Q_W,R_T,YT(:,t),PX(1:n_x,1:n_x,t-1),Ep);
    PX1=(PX1+PX1')/2;
    NIS_AGVI(:,t)   = NIS;
    EX(1:n_w,t)       = EX1(1:n_w);    % n = n_x*2 i.e., no of x + no of w
    PX(1:n_w,1:n_w,t) = PX1(1:n_w,1:n_w);
    %% Collecting W|y
    EX_wy   = EX1(end-n_x+1:end,1);
    PX_wy   = PX1(end-n_x+1:end,end-n_x+1:end,1);
    m       = triu(true(size(PX_wy)),1);  % Finding the upper triangular elements
    cwiwjy  = PX_wy(m)';                  % covariance elements between W's after update
    P_wy    = diag(PX_wy); % variance of updated W's
    i = 1; j = 1; k = 1;
    while i <= n_x-1
        PX_wy(i,j+1) = cwiwjy(k);
        j = j+1;
        k = k+1;
        if j == n_x
            i = i+1;
            j = i;
        end
    end
    PX_wy  = triu(PX_wy)+triu(PX_wy,1)';
    %%% 2nd Update step: 
    %% Computing W^2|y
    [m_wii_y,s_wii_y,m_wiwj_y, s_wiwj_y] = multiagvi.meanvar_w2pos(EX_wy,PX_wy,cwiwjy, P_wy, n_w2hat,n_x);

    %% Computing the covariance matrix \Sigma_W^2|y:
    cov_wijkl = multiagvi.cov_w2pos(PX_wy,EX_wy,n_w2,ind_cov,ind_mu);
%     cov_wpy   = cell2mat(reshape(cov_wijkl,size(cov_wijkl,2),1));
    %% Adding the variances and covariances of W^p to form \Sigma_W^p
    PX_wpy    = multiagvi.var_wpy(cov_wijkl,s_wii_y,s_wiwj_y);
    PX_wpy    = (PX_wpy+PX_wpy')/2;
    %% Creating E[W^p]
    EX_wpy        = [m_wii_y' m_wiwj_y];
    
    %% Computing prior mean and covariance matrix of Wp
    m_wsqhat    = E_P_old;
    s_wsqhat    = P_P_old;
    PX_wp       = multiagvi.covwp(P_P_old,m_wsqhat,n_x,n_w2,n_w);% old format
    
    %% Computing the prior covariance matrix \Sigma_W^p
    cov_prior_wijkl = multiagvi.priorcov_wijkl(m_wsqhat,ind_cov_prior,n_w2);
    %% Adding the variances and covariances for Prior \Sigma_W^p
    cov_wp              = cell2mat(reshape(cov_prior_wijkl,size(cov_prior_wijkl,2),1));
    s_wp                = zeros(size(PX_wp,1));
    s_wp(1:end-1,2:end) = cov_wp;
    PX_wp               = PX_wp + s_wp;
    PX_wp               = triu(PX_wp)+triu(PX_wp,1)'; % adding the lower triangular matrix
    PX_wp               = (PX_wp+PX_wp')/2;
    %% Creating Prior E[W^p]
    E_wp        = m_wsqhat;
    %% Smoother Equations
    [ES,PS] = multiagvi.agviSmoother(E_wp,s_wsqhat,PX_wp,EX_wpy,PX_wpy,E_P_old,P_P_old);
    EX(end-n_w2+1:end,t)                 = ES;
    PX(end-n_w2+1:end,end-n_w2+1:end,t)  = PS;
    
    E_Pw_y  = ES;
    P_Pw_y  = PS;
    P_Pw_y  = (P_Pw_y+P_Pw_y')/2;
    if any(eig(P_Pw_y)<0)
        P_PW_y = multiagvi.makePSD(P_Pw_y);
    end
    %% Converting from Pw to Lw
    [EL_pos,PL_pos] = multiagvi.PtoL(Cov_L_P_old,E_P_old,P_P_old,E_Lw(:,t-1),P_Lw(:,:,t-1),E_Pw_y,P_Pw_y);
    PL_pos      = (PL_pos+PL_pos')/2;
    if any(eig(PL_pos)<0)
        P_PW_y = multiagvi.makePSD(PL_pos);
    end
    E_Lw(:,t)   = EL_pos;
    P_Lw(:,:,t) = PL_pos;
end

runtime = toc(start)
% E_Lw(:,end)
Avg_NIS_time    = NIS_AGVI;
N_ANIS(p)          = length(Avg_NIS_time(Avg_NIS_time > 12.833 | Avg_NIS_time < 0.831));
disp(N_ANIS)
ANIS            = mean(Avg_NIS_time);
% figure;
% plot(1:T-1,Avg_NIS_time(2:end),'k');hold on; plot(1:T-1,5.02*ones(1,T-1),'r');hold on; plot(1:T-1,0*ones(1,T-1),'g')
% title('NIS');
% ylabel('test statistic');
end
mean_ANIS = sum(N_ANIS)/5
%% Plotting
disp_plot = 0;
if disp_plot==1
%  Plotting Variances
    E_W = zeros(total,length(EX));
    V_W = zeros(total,length(EX));
    for t = 1:length(EX)
        mL_old     = E_Lw(:,t);
        SL_old     = P_Lw(:,:,t);
        [mL, SL]   = multiagvi.convertstructure(mL_old,SL_old,n_x); %converted to new structure
        %% Matrices for mL and SL
        mL_mat = triu(ones(n_x));
        mL_mat(logical(mL_mat)) = mL;
        SL_mat = triu(ones(n_x));
        SL_mat(logical(SL_mat)) = diag(SL);
        
    %     m      = logical(triu(mL_mat));
        %% Initialization for Product L^T*L
        [E_P_new,P_P_new]         = multiagvi.LtoP(n_x,mL_mat,SL_mat,SL,ind_covij);
        P_P_new  = (P_P_new + P_P_new')/2;
        [E_W(:,t),P_P(:,:,t)]     = multiagvi.convertback(E_P_new,P_P_new,[],n_x);
        P_P(:,:,t) = (P_P(:,:,t)+P_P(:,:,t)')/2;
        V_W(:,t)                  = diag(P_P(:,:,t));
    end
    t  = 1:length(EX);
    for i=1:n_x
        figure;
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
%         ylim([0,5])
        title('variance')
    end
    % Plotting Covariances
    for i=1:total-n_w
        figure;
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
%         ylim([-2,2])
        title('covariance')
    end
end
