import mpi4py.MPI as mpi

comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
   print(f'Number of process: {size}')

comm.Barrier()

print(f'Hello from process: {rank}')