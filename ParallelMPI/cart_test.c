#include <stdio.h>
#include <mpi.h>
#include <math.h>

typedef struct {
  int n;
  int m;
  double alpha;
  double relax;
  double tol;
  int mits;
} parameters;

int main(int argc, char* argv[]){
	MPI_Init(&argc, &argv);
  /* Cartesian Grid Creation */
	int comm_size, my_world_rank, dim_size[2], periods[2];
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	if(comm_size != 80){
		dim_size[1] = dim_size[0] = sqrt(comm_size);
	}
	periods[0] = periods[1] = 0;
	MPI_Comm new_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_size, periods, 1, &new_comm);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank);
  /***************************/
  
  /* Input from process 0 */
  parameters param;
  /* Creating type to broadcast all parameters to all processes */ 
  int array_of_lengths[6] = {1, 1, 1, 1, 1, 1};
  MPI_Datatype array_of_types[6] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
  MPI_Aint array_of_displacements[6];
  array_of_displacements[0] = offsetof(parameters, n);
  array_of_displacements[1] = offsetof(parameters, m);
  array_of_displacements[2] = offsetof(parameters, alpha);
  array_of_displacements[3] = offsetof(parameters, relax);
  array_of_displacements[4] = offsetof(parameters, tol);
  array_of_displacements[5] = offsetof(parameters, mits);
  /*MPI_Aint lower_bound, extent_int, extent_double;
  MPI_Type_get_extent(MPI_INT, &lower_bound, &extent_int);
  array_of_displacements[1] = extent_int;
  array_of_displacements[2] = 2*extent_int;
  MPI_Type_get_extent(MPI_DOUBLE, &lower_bound, &extent_double);
  array_of_displacements[3] = 2*extent_int + extent_double;
  array_of_displacements[4] = 2*extent_int + 2*extent_double;
  array_of_displacements[5] = 2*extent_int + 3*extent_double; */
  MPI_Datatype broadcast_type;
  MPI_Type_create_struct(6, array_of_lengths, array_of_displacements, array_of_types, &broadcast_type);
  MPI_Type_commit(&broadcast_type);
  if(my_world_rank == 0){
    scanf("%d,%d", &(param.n), &(param.m));
    scanf("%lf", &(param.alpha));
    scanf("%lf", &(param.relax));
    scanf("%lf", &(param.tol));
    scanf("%d", &(param.mits));
  }
  MPI_Bcast(&param, 1, broadcast_type, 0, new_comm);
  MPI_Type_free(&broadcast_type);
  /************************/
  
  double xRight = 1.0;
  double xLeft = -1.0;
  double yBottom = -1.0;
  double yUp = 1.0;
  
  if(comm_size != 80){
    double root = sqrt(comm_size);
    double length = xRight - xLeft;
    double xLeft_local, xRight_local;
    double yBottom_local, yUp_local;
    // if(my_world_rank == 0){
      // int i, j, coords[2], curr_rank;
      // for(i = 0; i < sqrt(comm_size); i++){
        // for(j = 0; j < sqrt(comm_size); j++){
          // coords[0] = i;
          // coords[1] = j;
          // MPI_Cart_rank(new_comm, coords, &curr_rank);
          // xLeft_local = xLeft + ((double)coords[0])*(length/root);
          // xRight_local = xLeft_local + (length/root);          
          // yUp_local = yUp - ((double)coords[1])*(length/root);
          // yBottom_local = yUp_local - (length/root);
          // printf("Process with cart. coords (%d, %d) is %d in the world comm.\n", coords[0], coords[1], curr_rank);
          // printf("This process will solve in [%.5lf, %.5lf] x [%.5lf, %.5lf].\n", xLeft_local, xRight_local, yBottom_local, yUp_local);
        // }
      // }
    // }
  }
  MPI_Finalize();
	return 0;
}
