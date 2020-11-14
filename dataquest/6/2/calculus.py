# Function
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 3, 100)
y = -1*(x**2) + x*3 - 1
plt.plot(x,y)

def slope(x1,x2,y1,y2):
    rc = (y2 - y1)/(x2 - x1)
    return rc
slope_one = slope(0,4,1,13)
slope_two = slope(5,-1,16,-2)

print(slope_one)
print(slope_two)

# Symbolic math
import sympy
x2,y = sympy.symbols('x2 y')
limit_one = sympy.limit((-x2**2 +3*x2-1+1)/(x2-3) , x2, 2.9)
print(limit_one)

x,y = sympy.symbols('x y')
y = 3*(x**2) + 3*x - 3
limit_two = sympy.limit(y, x, 1)
print(limit_two)

y = x**3 + 2*(x**2) - x*10
limit_three = sympy.limit(y, x, -1)
print(limit_three)