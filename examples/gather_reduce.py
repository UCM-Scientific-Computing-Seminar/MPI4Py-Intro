import mpi4py.MPI as mpi
import numpy

comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Gather
send = numpy.array([rank, rank**2])
recv = None
if rank == 0:
   recv = numpy.empty((size,2), dtype=numpy.int64)

comm.Gather(send, recv, root=0)
print(f'Process {rank}: recv = {recv}')


comm.Barrier()
# Reduce
sum = numpy.array([0, 0])
comm.Reduce(send, sum, op=mpi.SUM, root=0)
print(f'Process {rank}: sum = {sum}')