from sympy import *
x, t, z, nu = symbols('x t z nu')

init_printing(use_unicode=True)

differential = diff(sin(x)*exp(x), x)
print(differential)

differential = diff(cos(x)*exp(x), x)
print(differential)

integral = integrate(exp(x)*sin(x) + exp(x)*cos(x), x)
print(integral)

integral = integrate(sin(x**2), (x, -oo, oo))
print(integral)

A = Matrix([[1, 2], [2, 2]])
eigenvalues = A.eigenvals()
print(f"matrix: {A} eigenvalues: {eigenvalues}")

print(latex(Integral(sqrt(1/x), x)))

expression = (x**2+2*x+1)
facor = factor(x**2+2*x+1)
print(f"expression: {expression} facotr: {factor}")


a= factor(x**2+2*x+1)
print(factor(x**3 - x**2 + x - 1),f"{a}")
