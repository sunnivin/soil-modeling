# from math import pi
# from pathlib import Path

# import numpy as np
# from matplotlib import pyplot as plt

# PLOT_FOLDER = "plots"

# u=150.     #x-position of the center
# v=75    #y-position of the center
# a=300.     #radius on the x-axis
# b=150    #radius on the y-axis

# t = np.linspace(0, 2*pi, 100)
# plt.plot( u+a*np.cos(t) , v+b*np.sin(t) )
# plt.grid(color='lightgray',linestyle='--')
# fig_name = Path.cwd()/PLOT_FOLDER/"ellipse.png"
# plt.savefig(fig_name)

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sympy import *

x,y = symbols('x y')

plot_implicit(Eq(x**2+y**2, 4))