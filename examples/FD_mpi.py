from time import time

import matplotlib.pyplot as plt
from numpy import sin, exp, pi, linspace, zeros_like, concatenate
import mpi4py.MPI as mpi
from math import ceil

u_exact = lambda t, x: exp(-t) * sin(x) + exp(-9*t) * sin(3*x)

comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
topo = comm.Create_cart([size], periods=False)

def heat_1d(Nt, Nx, t_end):
    local_size = ceil((Nx) / size)
    if rank == size - 1:
        local_size = Nx - local_size * rank

    # Problem parameters
    dx = 2 * pi / (Nx-1)
    dt = t_end / Nt
    x = linspace(rank * local_size * dx, ((rank+1) * local_size - 1) * dx, local_size)
    u = u_exact(0, x)

    # Time stepping
    for i in range(Nt):
        u = u + dt * u_xx(u,dx)

    return u

def u_xx(u, dx):
    bc = topo.neighbor_alltoall([u[0], u[-1]])
    u_xx = zeros_like(u)

    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
    if bc[0] is not None:
        u_xx[0] = (u[1] - 2*u[0] + bc[0]) / dx**2
    if bc[1] is not None:
        u_xx[-1] = (bc[1] - 2 * u[-1] + u[-2]) / dx ** 2

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
    glb_u = comm.gather(u, root=0)

    if rank == 0:
        glb_u = concatenate(glb_u)
        plot_sol(glb_u,t_end)
