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
  /* A test to see how cartesian and world communicator coordinates match */
  // if(my_world_rank == 0 && comm_size != 80){
    // int i, j, coords[2], curr_rank;
    // for(i = 0; i < sqrt(comm_size); i++){
      // for(j = 0; j < sqrt(comm_size); j++){
        // coords[0] = i;
        // coords[1] = j;
        // MPI_Cart_rank(new_comm, coords, &curr_rank);
        // printf("Process with cart. coords (%d, %d) is %d in the world comm.\n", coords[0], coords[1], curr_rank);
      // }
    // }
  // }
  /*************************************************************************/
  double xRight = 1.0;
  double xLeft = -1.0;
  double yBottom = -1.0;
  double yUp = 1.0;
  
  if(comm_size != 80){
    double root = sqrt(comm_size);
    double length = xRight - xLeft;
    double xLeft_local, xRight_local;
    double yBottom_local, yUp_local;
    if(my_world_rank == 0){
      int i, j, coords[2], curr_rank;
      for(i = 0; i < sqrt(comm_size); i++){
        for(j = 0; j < sqrt(comm_size); j++){
          coords[0] = i;
          coords[1] = j;
          MPI_Cart_rank(new_comm, coords, &curr_rank);
          xLeft_local = xLeft + ((double)coords[1])*(length/root);
          xRight_local = xLeft_local + (length/root);          
          yUp_local = yUp - ((double)coords[0])*(length/root);
          yBottom_local = yUp_local - (length/root);
          printf("Process with cart. coords (%d, %d) is %d in the world comm.\n", coords[0], coords[1], curr_rank);
          printf("This process will solve in [%.5lf, %.5lf] x [%.5lf, %.5lf].\n", xLeft_local, xRight_local, yBottom_local, yUp_local);
        }
      }
    }
  }
  MPI_Finalize();
	return 0;
}
