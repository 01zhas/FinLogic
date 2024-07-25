import cvxpy as cp
import numpy as np

# Определите простую задачу оптимизации
x = cp.Variable(integer=True)
y = cp.Variable()

objective = cp.Minimize((x - 1)**2 + (y - 2.5)**2)
constraints = [x + y == 1, x >= 0, y >= 0]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS_BB, verbose=True)

print(f"Статус решения: {problem.status}")
print(f"Оптимальное значение переменных: x = {x.value}, y = {y.value}")