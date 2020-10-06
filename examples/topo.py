import mpi4py.MPI as mpi

comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

ndim = 2
dims = mpi.Compute_dims(size, [0]*ndim)

topo = comm.Create_cart(dims, periods=False)

print(f'Process {rank}: coordinate = {topo.coords}, neighbor = {topo.outedges}')
comm.Barrier()

recvbuf = topo.neighbor_alltoall([rank]*4)
print(rank, "->", recvbuf)