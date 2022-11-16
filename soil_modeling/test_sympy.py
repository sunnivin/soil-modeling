from sympy import symbols, factor, expand 

x, y = symbols('x y')

expression = x + 2*y 
print(f"Expression {expression}")

expanded_expr = expand(x*expression)

print(f"expaneded expression {expanded_expr}")

facor_expr = factor(expanded_expr)
print(f"facorized expanded expression {facor_expr}")