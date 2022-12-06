clear;

%#######################################################
X = [0.2:0.2:4];
disp([X; sin(X)]');

%#######################################################
cirle_area = @(r) pi * r^2;
cirle_area_n =  cirle_area(1)

%#######################################################
test_N = randn(3, 5);
test_n = 2;
cropped_N= top_right(test_N, test_n)


%#######################################################
test_x = 7;
disp(ifelselogic(test_x))

%#######################################################
test_matrix = ones(8);
res = fordiag(test_matrix)


%#######################################################
test_a = 121; test_b = 5; test_c = 911;
%test_maxof3 = max([test_a, test_b, test_c]);
maxof3 = ifelsemax(test_a, test_b, test_c)

%#######################################################
test_lhs = 15;
test_rhs = 15;
test_op = "*";
result = switchcase(test_lhs, test_op, test_rhs)


%#######################################################
syms x y t h g R;

simplify((x + y) / (1/x + 1/y))

eq = 4* t * h^2 + 20 * t - 5 * g;

eq_t = solve(eq == 0, t)

eq_g = solve(eq == 0, g)

syms x y R;
cirle_and_line_eq = solve( (x- 2)^2 + (y-4)^2 - R^2 == y - x/2 - 1, [x, y]);
disp(cirle_and_line_eq.x);
disp(cirle_and_line_eq.y);

%#######################################################
syms x(t) s(a) y(t);
ode1_ = dsolve(diff(diff(x, t), t) + 2 * diff(x, t) + x == 0)
ode3_ = dsolve(diff(y, t)+ 4 *y == 60,  y(0)==5 )

syms x a s(a);
ode2_ = dsolve(diff(s, a) == - a * x^2 )


%#######################################################
syms y(t);
tspan = [0; 10];
y0 = 2;
[t,y] = ode45(@(t,y) 3*cos(y) + sin(t), tspan, y0);
disp("solution");

plot(t,y,'-o')
