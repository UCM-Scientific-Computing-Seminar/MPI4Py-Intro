import mpi4py.MPI as mpi
import numpy

comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = None
if rank == 0:
    data = numpy.array([1,2,3])
    print(f'Process 0 is sending: {data} to process 1')
    comm.Send(data, dest=1)
elif rank == 1:
    data = numpy.empty(3, dtype=numpy.int64)
    comm.Recv(data, source=0)

comm.Barrier()
print(f'Process {rank}: data = {data}')