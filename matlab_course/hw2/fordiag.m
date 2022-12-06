function M = fordiag(A)
  n = length(A(:, 1));
  m = length(A(1, :));
  for i = 1:n
    for j = 1:m
      if i == j || i+j == n + 1
        M(i, j) = A(i, j);
      else
        M(i, j) = 0;
      end
    end
  end
end
