#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char* argv[]){
	MPI_Init(&argc, &argv);
	int comm_size, my_world_rank, dim_size[2], periods[2];
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	if(comm_size != 80){
		dim_size[1] = dim_size[0] = sqrt(comm_size);
	}
	periods[0] = periods[1] = 0;
	MPI_Comm new_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_size, periods, 1, &new_comm);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank);
  if(my_world_rank == 0 && comm_size != 80){
    int i, j, coords[2], curr_rank;
    for(i = 0; i < sqrt(comm_size); i++){
      for(j = 0; j < sqrt(comm_size); j++){
        coords[0] = i;
        coords[1] = j;
        MPI_Cart_rank(new_comm, coords, &curr_rank);
        printf("Process with cart. coords (%d, %d) is %d in the world comm.\n", coords[0], coords[1], curr_rank);
      }
    }
  }
  MPI_Finalize();
	return 0;
}
