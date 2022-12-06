function r = switchcase(lhs, op, rhs)
    switch op
      case "+"
        r = lhs + rhs;
      case "*"
        r = lhs * rhs;
      case "-"
        r = lhs - rhs;
      case "/"
        r = lhs / rhs;
    end
end
