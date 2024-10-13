% Main script for the McCall job search problem

% Housekeeping
clear all

% Enter model parameters
N     =  100;                 % wage grid size
beta  =  .9;                  % discount factor
w_min =  50;                  % minimum wage
w_max =  150;                 % maximum wage
c     =  [.34 .67]';          % replacement rate

% Construct state variable
w = linspace(w_min,w_max,N)';     % the wage grid 

% Solve model for each parameter configuration
V  = zeros(N,2);
x  = zeros(N,2);
Pi = zeros(2,N,N); 
for i = 1:2
  [V(:,i),x(:,i),Pi(i,:,:)]   = McCall_ddp(beta,c(i),w);
end

% identify the reservation wages
slope = diff(V)./repmat(diff(w),1,2); % numerical slope
dslope = diff(slope);                 % difference in numerical slope
[value,index] = max(dslope);          % index of the max difference
w_res         = w(index+1);           % correct index for the differencing
  
% compute elasticitiy of the reservation wage
w_ela = (diff(w_res)./w_res(1:end-1,:))./(diff(c)./c(1:end-1,:));
  
% Plot value function of the unemployed worker
figure(2);
subplot(1,1,1);
plot(w,V(:,1),'k-',w,V(:,2),'r--');
ylabel('V(w)');
xlabel('w');
title('Value Function') ;
leg{1} = ['c = ',num2str(c(1))] ;
leg{2} = ['c = ',num2str(c(2))] ;
legend(leg);