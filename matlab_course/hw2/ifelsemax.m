function m = ifelsemax(a, b, c)
      if a >= b
        if a >= c
          m = a
        else
          m = c
        end
      elseif b >= c
        if b >= a
          m = b
        else
          m = a
        end
      elseif c >= a
        if c >= b
          m = c
        else
          m = b
        end
      end
end
