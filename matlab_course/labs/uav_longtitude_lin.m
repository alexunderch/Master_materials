function uav_longtitude_lin

A = [-0.543 -0.5332 0 -9.7295;
    -2.7791 -10.3435 8.51 -1.1732;
    -0.3404 -2.0302 0. 0.;
    0. 0. 1. 0.];

B = [2.4224 0.0224;
    -20.2054 0.;
    -18.4384 0.;
    0. 0.];

C = [1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1];

D = [0 0;
    0 0;
    0 0;
    0 0];

states = {'u', 'omega', 'q', 'theta'}; 
inputs = {'delta_e', 'delta_t'};                      
outputs = {'u', 'omega', 'q', 'theta'};

sys_ss = ss(A,B,C,D,'statename',states,'inputname',inputs,'outputname',outputs);

fprintf('%s\n', 'Eigenvalues of the uncontrolled system');
poles = eig(A) %fing eigenvalues of the matrix A

fprintf('%s\n', 'Is the system controllable (rg(G) = 4)');
cm = ctrb(sys_ss);
controllabitlity = rank(cm)


%finite time
Q = C' * C;
F = Q;
rho = 0.9;
R = rho* (D' * D + eye(2));  %positively defined
tspan = 0:0.01:10;

function rhs_vect = Riccati_rhs_vect(t, p)
    P = reshape(p, 4, 4);
    rhs_vect = reshape(P*A + A'*P - P*B*(R^-1)*B'*P + Q, 16, 1);
end

p_ICs = reshape(F, 16, 1);
[~, p_evolution] = ode45(@Riccati_rhs_vect, tspan, p_ICs);
P_finite_hor_sol = [tspan', flipud(p_evolution)];

function rhs = cl_rhs(t, x, P_sol)
    p=interp1(P_sol(:, 1), P_sol(:, 2:end), t, "spline"); %interpolate to the full interval
    P=reshape(p, 4, 4);
    u=-inv(R)*B'*P*x;
    rhs = A*x + B*u;
end

x_ICs = [1, 0, -1, 0];
[~, x_finite_hor] = ode45(@(t, x)cl_rhs(t,x,P_finite_hor_sol), tspan, x_ICs);

figure('Name','Transition process with a finite horizon LQR control'); 
title('Transition process with a finite horizon LQR control');
plot(tspan, x_finite_hor);
legend('u', 'omega', 'q', 'theta');
figure('Name', 'Riccati solution in a finite horizon LQR problem'); % Открываем новое окно
plot(tspan, P_finite_hor_sol(:, 2:end)); % просмотр динамики всего вектора состяний
title('Riccati solution in a finite horizon LQR problem');


%----------infinite-----------
P_inf_hor = icare(A,B, Q, R);
K = (R^-1) * B'* P_inf_hor;
% K = lqr(A, B, Q, R);
Ac = (A-B*K);
Bc = B;
Cc = C;
Dc = D;


sys_cl = ss(Ac,Bc,Cc,Dc,'statename',states,'inputname',inputs,'outputname',outputs);
r = 2*ones(length(inputs), length(tspan));

[y, tspan, x_inf_time] = lsim(sys_cl, r, tspan, x_ICs);
figure('Name', 'Output along state feedback'); 

[AX,~,~] = plotyy(tspan,y(:,1),tspan,y(:,2),'plot');
set(get(AX(1),'Ylabel'),'String','V_x')
set(get(AX(2),'Ylabel'),'String','Qth angular velocity')
title('Transition process with an infinite horizon LQR control')

figure('Name', 'States along state feedback'); 
plot(tspan, x_inf_time); 
legend('u', 'omega', 'q', 'theta');

% % ---------observ--------------
states = {'u', 'omega', 'q', 'theta'}; 
inputs = {'delta_e', 'delta_t'};                      
outputs = {'u', 'q'};
C = [1 0 0 0;
     0 0 1 0];
Cc = C;
D = [0 0;
    0 0];

sys_ss = ss(A,B,C,D,'statename',states,'inputname',inputs,'outputname',outputs);
ob = obsv(sys_ss);
observability = rank(ob)
fprintf('%s\n', 'Poles for the controlled system');
poles = eig(Ac) %for closed system
pol = [-98 -32 -32 -33];
L = place(A', C', pol)';

Ace = [(A-B*K) (B*K);
       zeros(size(A)) (A-L*C)];
Bce = [B;
       zeros(size(B))];
Cce = [Cc zeros(size(Cc))];

Dce = [0 0;
       0 0];

states = {'u' 'omega' 'q' 'theta' 'e1' 'e2' 'e3' 'e4'};
inputs = {'delta_e', 'delta_t'};                      
outputs = {'u', 'q'};

sys_est_cl = ss(Ace, Bce, Cce, Dce,'statename',states,'inputname',inputs,'outputname',outputs);
% r = zeros(size(tspan));
e_ICs = [-1 2 1 0.5];
est_ICs = [x_ICs, e_ICs];
[est_y, tspan, est_xe] = lsim(sys_est_cl, r, tspan, est_ICs);
figure('Name', 'Output along state feedback');
[AX,~,~] = plotyy(tspan,est_y(:,1),tspan,est_y(:,2),'plot');
set(get(AX(1),'Ylabel'),'String','V_x')
set(get(AX(2),'Ylabel'),'String','Qth angular velocity')
title('Transition process with observer-based state-feedback Control')


figure('Name', 'States and observer errors vectors'); 
plot(tspan, est_xe); 
legend('u', 'omega', 'q', 'theta');

eig(Ace)

end