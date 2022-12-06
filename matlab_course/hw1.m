clear;

% ##########################################################
disp("task0`");
% a)
mult1 = (3 + 1/3 + 2.5) / (2.5 - 1 - 1/3);
mult2 = (4.6 - 2 - 1/3) / (4.6 + 2 + 1/3);
mult3 = 0.05 / (1/7 - .125) + 5.7;
expression_A = mult1 * mult2 / mult3;
expression_A

% b)
expression_B = (5.9 ^ 2  - 2.4 ^ 2) / 3.0 + (log10(12890)/ exp(0.3))^2;
expression_B

% c)
expression_C = sind(80)^2 - (cosd(14) * sind(80))^2 / nthroot(0.18, 3)
expression_C

% d)
expression_D = (2 * 10 ^ 4 / 3) * 5  + (1 / (2 * 10^(-2))) + factorial(6);
expression_D

disp("task1");
% ##########################################################
x = pi / 5;
isTrue =  cos(x/2)^2 - (tan(x) + sin(x)) / (2 * tan(x)) - 0. < 1e-5

disp("task2");
% ##########################################################
a = [1 2 3; 5 6 7];
b = [1 1 1; 0 4 5];
c = [3 1; 2 2; 0 0];
d = [4 4 2; -1 2 4];
e = [1 2 3 4];
f = [2 1 4 3];

disp("ab");
%adding matrices of the same shape
disp(a + b);
%adding a number to a matrix -- implicit broadcasting to the matrix shape
disp(a + 2);

disp("cd");
%shapes of matrices are complement  -- ok
disp(c * d);
disp(c * 2);

disp("ab");
%shapes of matrices are not complement  -- error
%disp(a * b);
%shapes of matrices are the same  -- ok
disp(a.* b);

disp("ef");
%shapes of matrices are not complement  -- error
%disp(e * f);
%scalar multiplication
disp(e * f');

disp("task3");
% ##########################################################
A = eye(2); B = ones(2); C = zeros(2);

D = [A B C; C B A];
D

disp("task4");
% ##########################################################
tmp = reshape([36: -2: 2], [6, 3])';
disp("the 2nd row");
disp(tmp(2, :));
disp("the 6th row");
disp(tmp(:, 6));
disp("the1st 2 elements of the 3rd row");
disp(tmp(3, 1:2));
disp("the last 2 elements of the 1st row");
disp(tmp(1, end-2:end));

disp("task5");
% ##########################################################
tmp_ = [1, 2, 3;2, 6, 4; 1, 2, round(rand(1)*5)];
disp("inverse");
inv(tmp_)
disp("transpose");
transpose(tmp_)

disp("task6");
% ##########################################################
% Enter as a string with numbers separated by commas or spaces
userResponse = input('enter the weights separated by commas or spaces ', 's');
userResponse = strrep(userResponse, ',', ' ');
w = sscanf(userResponse, '%f');

userResponse = input('enter the score separated by commas or spaces ', 's');
userResponse = strrep(userResponse, ',', ' ');
g = sscanf(userResponse, '%f');
disp("result");
dot(w, g) / sum(w)

disp("task7");
% ##########################################################
A_d = [-1 2 3;4 5 -6;7 -8 9];
B_d = [5;6;-7];
Q_d = [49 56 63; 56 64 72; 63 72 81];
R_d =1.;
disp("Lyapunov equation solution");
lyap(A_d, Q_d)

disp("Riccati equation solution");
are(A_d, B_d * inv(R_d) * B_d.', Q_d)
icare(A_d, B_d, Q_d, R_d)




