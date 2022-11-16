# Testing matrices and eigenvalues calculated with Sympy 
from sympy import * 

init_printing(use_unicode=True)

sigma = Matrix([[90, -30, 0], [-30, 120, -30], [0,-30,90]])

print(f"sigma: {sigma}")

# 1. Determine the stess variants --> Find the determant 
determinant = sigma.det()
print(f"determinant {determinant}")

eigenvalues = sigma.eigenvals()
print(f"eigenvalues {eigenvalues}")

eigenvectors = sigma.eigenvects()
print(f"eigenvectors {eigenvectors}")