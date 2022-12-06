function sr = ifelselogic(x)
  if x > 5
    if x > 10
      sr = "it's a big number";
    elseif x == 7
      sr = "it's a lucky number";
    else
      sr = "it's a number"
    end
  else
    sr = "it's a small number";
  end
end
