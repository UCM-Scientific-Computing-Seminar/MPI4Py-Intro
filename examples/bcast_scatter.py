import mpi4py.MPI as mpi
import numpy

comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Broadcasting
if rank == 0:
   data = numpy.array([1,2,3])
   print(f'Process 0 is sending: {data} to all processes')
else:
   data = numpy.empty(3, dtype=numpy.int64)

comm.Bcast(data, root=0)

comm.Barrier()
print(f'Process {rank}: data = {data}')



comm.Barrier()
# Scattering
numPerRank = 10
data = None
if rank == 0:
   data = numpy.arange(numPerRank*size)
   print(f'Process 0 is scattering: {data} over all processes')

recv = numpy.empty(numPerRank, dtype=numpy.int64)

comm.Scatter(data, recv, root=0)

comm.Barrier()
print(f'Process {rank}: data = {recv}')

