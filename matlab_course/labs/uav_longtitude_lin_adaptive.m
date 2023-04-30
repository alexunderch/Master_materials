function uav_longtitude_lin_adaptive

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

Q = zeros(4, 4);
Q(1, 1) = 5000;
Q(3, 3) = 100;

rho = 0.9;
R = rho* (D' * D + eye(2));

P_inf_hor = icare(A,B, Q, R);
K_LQR = (R^-1) * B'* P_inf_hor;

A_m = A - B * K_LQR;
B_m = B;

T = 5;
tspan = 0:0.1:T;

sys_ICs = [1, 0, -1, 0];
x_m_ICs = sys_ICs;
theta_ICs = [0, 0, 0, 0];
ICs = [sys_ICs, x_m_ICs, theta_ICs];

% g_ref = 0 -- stability
g_ref = @(t)[0; 0];
% g_ref = @(t)[sin(20*t)*cos(40*t); sin(20*t)*cos(40*t)];

sys_m = ss(A_m, B_m, eye(size(A, 1)), zeros(size(B_m)));
g_ref_vals = zeros(size(B_m, 2), length(tspan));

for i =1:size(tspan, 2)
    g_ref_vals(:, i) = g_ref(tspan(i));
end

x_m =lsim(sys_m, g_ref_vals, tspan, x_m_ICs); 
figure('Name', 'Reference model trajectories'); 
plot(tspan, x_m); 
legend('x_m', 'dx_m', 'fi_m', 'dfi_m');
% 
[t_plot, x_cl_adapt] = ode45(@(t, x)pendulim_on_cart_RHS(t, A, B, K_LQR, A_m, B_m,...
    x, g_ref, @MIMO_adaptive_control), tspan, ICs);

figure('Name','Extended system trajectories along adaptive conrol'); 
plot(t_plot, x_cl_adapt); % просмотр динамики всего вектора состяний
legend('x','dx', 'fi', 'dfi',...
    'x_m', 'dx_m', 'fi_m', 'dfi_m',...
    'Theta_1', 'Theta_2');

figure('Name','Tracking errors along adaptive conrol'); 
plot(t_plot, x_cl_adapt(:, 1:4) - x_cl_adapt(:, 5:8) );
legend('e_1', 'e_2', 'e_3', 'e_4');

[t_plot, x_cl_LQR] = ode45(@(t, x)pendulim_on_cart_RHS(t, A, B, K_LQR, A_m, B_m,...
    x, g_ref, @LQR_control), tspan, ICs);

figure('Name','Extended system trajectories along LQR conrol'); 
plot(t_plot, x_cl_LQR(:, 1:8));
legend('x','dx', 'fi', 'dfi',...
    'x_m', 'dx_m', 'fi_m', 'dfi_m');

figure('Name','Tracking errors along LQR conrol'); 
plot(t_plot, x_cl_LQR(:, 1:4) - x_cl_LQR(:, 5:8) );
legend('e_1', 'e_2', 'e_3', 'e_4');

function dxdt = pendulim_on_cart_RHS(t, A, B, K_LQR, A_m, B_m, x, g_ref, u_ref)
dxdt = zeros(size(x)); 

cur_g_ref = g_ref(t); 
% Размерности системы
n = 4; % размерность вектора состояния
r = 2; % размерность вектора управления
l = 2; % размерность вектора неопределенных параметров

% Индексы для элементов подсистем в расширенном векторе
sys_x_indexes = 1:n;
x_m_indexes = (n+1) : (2*n);
hat_Theta_indexes = (2*n+1) : (2*n + l*r); 

sys_x = x(sys_x_indexes); % выделяем вектор состояния системы
x_m = x(x_m_indexes);     % выделяем вектор ЭМ
% Выделяем вектор элементов матрицы Theta.
% Используем транспонирования для правильного 
% восставновления матрицы hat_Theta.

hat_Theta = reshape(x(hat_Theta_indexes), r,l)'; 

% Задаем неопределенное воздействие w. 
% w(x) = Theta^T * Phi(x) = [ -k1; -k2 ]^T * [ x2; x4 ]; 
% k1 > 0; k2 > 0; 
k1 = 10; k2 = 10; % неизвестные коэффициенты
Theta = [ -k1; -k2 ];

Phi = [ sys_x(2); sys_x(4) ]; 
w = Theta' * Phi; 

if isequal(u_ref, @MIMO_adaptive_control)
   u = MIMO_adaptive_control(K_LQR, sys_x, r, hat_Theta, Phi, cur_g_ref);
else
   u = LQR_control(K_LQR, sys_x);
end

dxdt(sys_x_indexes) = A*sys_x + B*(u+w);
dxdt(x_m_indexes) = A_m*x_m + B_m*cur_g_ref;

e = sys_x - x_m;
d_hat_Theta = get_d_hat_Theta(n, l, A_m, B, Phi, e);
% Правая часть для матрицы Theta
dxdt(hat_Theta_indexes) = reshape(d_hat_Theta', [], 1);
end

function u = LQR_control(K_LQR, sys_x)
    u =  -K_LQR * sys_x;
end     

% Фунция адаптивного управления 
function u = MIMO_adaptive_control(K_LQR, sys_x, r, hat_Theta, Phi, cur_g_ref)
K_x = -K_LQR; 
K_g = eye(r);
u = K_x * sys_x + K_g * cur_g_ref - hat_Theta' * Phi;
end

% Функция адаптации 
function d_hat_Theta = get_d_hat_Theta(n, l, A_m, B, Phi, e)
G = eye(l);
Q = eye(n);
P = lyap(A_m, Q);

d_hat_Theta = G * Phi * e' * P * B;
end

end