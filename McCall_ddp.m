% Function that uses ddpsolve to solve the McCall job search problem
%
% Inputs:
%   beta    - discount factor
%   c       - replacement rate
%   w       - wage grid

function [V,x,Pi] = McCall_ddp(beta,c,w)
  N = length(w);
  reject = 1;                       % indicates reject wage offer
  accept = 2;                       % indicates accept wage offer
  
% Construct reward function f(w,x)
  f = zeros(N,2); 
  f(:,accept) = w;                  % gets the wage
  f(:,reject) = c*w;                % gets the unemployment benefits 

% Construct state transition probability matrix P(w'|w,x)
  P             = zeros(2,N,N);
  P(reject,:,:) = (1/N)*ones(N,N);  % uniform i.i.d distribution
  P(accept,:,:) = eye(N,N);         % keeps job
  
% Pack model structure
  model.reward     = f;
  model.transprob  = P;
  model.discount   = beta;

% Solve model
  [V,x,Pi] = ddpsolve(model);
end