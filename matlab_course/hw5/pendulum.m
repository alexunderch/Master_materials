clear all;

gamma = 0.05;
g = 10;
L = 0.1;
M = 0.1;

A = [0, 1; g/L, -gamma/(M*L)];
B = [0; 1];
Q = [5, 1; 1, 5];
R = 1;


x0 = [pi; 1];
P = icare(A, B, Q, R);

%test
% sys = ss(A, B, [1 1], 1);
% K = lqr(sys, Q, R)

K = inv(R) * B' * P;