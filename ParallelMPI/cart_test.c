#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char* argv[]){
	MPI_Init(&argc, &argv);
	int comm_size, dim_size[2], periods[2];
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	if(comm_size != 80){
		dim_size[1] = dim_size[0] = sqrt(comm_size);
	}
	periods[0] = periods[1] = 0;
	MPI_Comm new_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_size, periods, 1, &new_comm);
	MPI_Finalize();
	return 0;
}
