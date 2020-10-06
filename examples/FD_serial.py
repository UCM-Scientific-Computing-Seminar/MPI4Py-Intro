from time import time

import matplotlib.pyplot as plt
from numpy import sin, exp, pi, linspace, zeros_like, concatenate

u_exact = lambda t, x: exp(-t) * sin(x) + exp(-9*t) * sin(3*x)

def heat_1d(Nt, Nx, t_end):
    # Problem parameters
    dx = 2 * pi / (Nx-1)
    dt = t_end / Nt
    x = linspace(0, 2*pi, Nx)
    u = u_exact(0, x)

    # Time stepping
    for i in range(Nt):
        u = u + dt * u_xx(u,dx)

    return u

def u_xx(u, dx):
    u_xx = zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2

    return u_xx

def plot_sol(u, t_end):
    x = linspace(0, 2*pi, u.size)
    plt.plot(x, u_exact(0, x))
    plt.plot(x, u)
    plt.plot(x, u_exact(t_end,x))
    plt.legend(['Initial solution', 'FD solution', 'Exact solution'])
    plt.show()

    plt.plot(x, u - u_exact(t_end,x))
    plt.show()
if __name__ == '__main__':
    Nt = 100_000
    Nx = 2_000
    t_end = 0.1
    u = heat_1d(Nt, Nx, t_end)
    plot_sol(u,t_end)
