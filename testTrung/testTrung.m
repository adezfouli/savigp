clear all; clc;
% tests Trung's AVIGP code using a MoDG approximation

%% Loading data and general settings
path(genpath('/Users/ebonilla/Documents/research/projects/avigp/agp'), path());
data = load('data.csv');
x = data(1,:)'; y = data(2,:)';
plot(x,y, '.', 'MarkerSize', 12);
rng(1110, 'twister');
[N,D]       = size(x);
Q           = 1; % no. latent functions
K           = 1; % no. mixture components
SIGMA_S     = 0.5; % (initial) Hyperparameter 
L_I         = 0.2; %      values 
SIGMA_N     = 0.2; % (initial) likelihood variance
LEARN_HYPER = 0;   % learn hyper-parameters
LEARN_LIK   = 0;   % learn likelihood parameters
%
POSTMEAN0   = 0;   % intialisation of posterior mean 
POSTCOV0    = 0.5; %      and posterior covariance


%% Sets data and parameters
m.x = x; m.y = y; m.N = N; m.Q = Q; m.K = K;
m.pars.M = POSTMEAN0*ones(N*Q,K);
m.pars.L = log(sqrt(POSTCOV0)*ones(N*Q,K));

%% pre-processing
% m.mean_y = mean(y);
% m.y      = m.y - m.mean_y;

%% covariance hyperparameters
m.pars.hyp.covfunc = @covSEard;
m.pars.hyp.cov = cell(Q,1);
m.pars.hyp.cov{1} = log([sqrt(L_I)*ones(D,1); sqrt(SIGMA_S)]);
m.pars.w = log(1/K)*ones(K,1);

%% Likelihood function 
m.likfunc = @llhGaussian;
m.pars.hyp.likfunc = m.likfunc;
m.pred = @mixturePredRegression;
m.pars.hyp.lik = log(sqrt(SIGMA_N));


%% configurations
conf.nsamples               = 10000;
conf.covfunc                = @covSEard;
conf.maxiter                = 100;
conf.displayInterval        = 10;
conf.checkVarianceReduction = false;
conf.latentnoise            = 0;
conf.learnhyp               = LEARN_HYPER;
conf.learnlik               = LEARN_LIK;

%% Optimisation structures for all parameters
opts = struct('Display','iter','Method','lbfgs','MaxIter',5,...
    'MaxFunEvals',100,'DerivativeCheck','off');
conf.varopts = opts;   conf.varopts.Maxiter = 100; % variational parametes
conf.hyperopts = opts; % hyperparameters
conf.likeopts = opts; % likelihood parameters

    
%% Learning
tic;
m = learnMixtureGaussians(m,conf);
toc

%% post-processing
% m.pars.M = m.pars.M + m.mean_y;


%% Posterior 
figure; hold on;
mu    = m.pars.M;
sigma = exp(2*m.pars.L);
plotMeanAndStd(m.x,mu,2*sqrt(sigma),[7 7 7]/8);
plot(m.x, m.y, 'o'); title('Posterior');


%% Predictions
range = (min(m.x) - 0.01 : 0.01 : max(m.x) + 0.01)';
[fmu,~,yvar] = feval(m.pred, m, conf, range);
figure; hold on;
plotMeanAndStd(range,fmu,2*sqrt(yvar),[7 7 7]/8);
plot(m.x, m.y, 'o'); title('Predictions');

fprintf('Final ELBO = %.4f\n', m.fval(end));
