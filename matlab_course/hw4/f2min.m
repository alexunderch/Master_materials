function f = f2min(x)
    k = 1:10;
    f = 2 + 2*k - exp(k * x(1)) - exp(k * x(2)); 
end