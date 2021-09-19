#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

typedef struct {
  int n;
  int m;
  double alpha;
  double relax;
  double tol;
  int mits;
  double deltaX;
  double deltaY;
} parameters;

int main(int argc, char* argv[]){
	MPI_Init(&argc, &argv);
  /* Cartesian Grid Creation */
	int comm_size, my_world_rank, dim_size[2], periods[2];
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	if(comm_size != 80){
		dim_size[1] = dim_size[0] = sqrt(comm_size);
	}else{
        dim_size[0] = 8; // rows
        dim_size[1] = 10; // columns
    }
	periods[0] = periods[1] = 0;
	MPI_Comm new_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_size, periods, 0, &new_comm);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank);
  /***************************/
  
  /* Input from process 0 */
  parameters param;
  /* Creating type to broadcast all parameters to all processes */ 
  int array_of_lengths[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  MPI_Datatype array_of_types[8] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_DOUBLE, MPI_DOUBLE};
  MPI_Aint array_of_displacements[8];
  array_of_displacements[0] = offsetof(parameters, n);
  array_of_displacements[1] = offsetof(parameters, m);
  array_of_displacements[2] = offsetof(parameters, alpha);
  array_of_displacements[3] = offsetof(parameters, relax);
  array_of_displacements[4] = offsetof(parameters, tol);
  array_of_displacements[5] = offsetof(parameters, mits);
  array_of_displacements[6] = offsetof(parameters, deltaX);
  array_of_displacements[7] = offsetof(parameters, deltaY);

  MPI_Datatype broadcast_type;
  MPI_Type_create_struct(8, array_of_lengths, array_of_displacements, array_of_types, &broadcast_type);
  MPI_Type_commit(&broadcast_type);
  if(my_world_rank == 0){
    scanf("%d,%d", &(param.n), &(param.m));
    scanf("%lf", &(param.alpha));
    scanf("%lf", &(param.relax));
    scanf("%lf", &(param.tol));
    scanf("%d", &(param.mits));
  }
  param.deltaX = (2.0)/(param.n-1);
  param.deltaY = (2.0)/(param.m-1);
  MPI_Bcast(&param, 1, broadcast_type, 0, new_comm);
  MPI_Type_free(&broadcast_type);
  /************************/
  
  /* Local parameters calculation */
  double xRight = 1.0;
  double xLeft = -1.0;
  double yBottom = -1.0;
  double yUp = 1.0;
  double xLeft_local, xRight_local, yBottom_local, yUp_local;
  int n_local, m_local, coords[2];
  double *u_local, *u_old_local, *fXsquared, *fYsquared;

  if(comm_size != 80){
    double root, length;
    root = sqrt(comm_size);
    length = xRight - xLeft;
    MPI_Cart_coords(new_comm, my_world_rank, 2, coords);
    xLeft_local = xLeft + ((double)coords[1])*(length/root);
    xRight_local = xLeft_local + (length/root);          
    yUp_local = yUp - ((double)coords[0])*(length/root);
    yBottom_local = yUp_local - (length/root);
    n_local = param.n/root;
    m_local = param.m/root;
  }else{
    n_local = 84; // We have decided we have divided x axis in 10 subspaces
    m_local = 105; // We have decided to divide y axix in 8 subspaces
    MPI_Cart_coords(new_comm, my_world_rank, 2, coords);
    xLeft_local = xLeft + ((double)coords[1])*0.2;
    xRight_local = xLeft_local + 0.2;          
    yUp_local = yUp - ((double)coords[0])*0.25;
    yBottom_local = yUp_local - 0.25; 
  }
  u_local = (double*)calloc((n_local + 2)*(m_local + 2), sizeof(double));
  u_old_local = (double*)calloc((n_local + 2)*(m_local + 2), sizeof(double));
  fXsquared = (double*)calloc(n_local, sizeof(double));
  fYsquared = (double*)calloc(m_local, sizeof(double));

  if(u_local == NULL || u_old_local == NULL || fXsquared == NULL || fYsquared == NULL){
    if(my_world_rank == 0){
      printf("Not enough memory for 2 %ix%i matrices, plus one 1x%i and one 1x%i vectors.\n", n_local + 2, m_local + 2, n_local, m_local);
    }
    MPI_Finalize();
    exit(1);
  }
  /********************************/

  /* Neighbor calculation */
  int north, south, east, west, neigh_coords[2];
  // Calculation of north neighbor
  neigh_coords[0] = coords[0] - 1;
  neigh_coords[1] = coords[1];
  if(neigh_coords[0] < 0){
    north = MPI_PROC_NULL;
  }else{
    MPI_Cart_rank(new_comm, neigh_coords, &north);
  }
  // Calculation of south neighbor
  neigh_coords[0] = coords[0] + 1;
  neigh_coords[1] = coords[1];
  if(neigh_coords[0] > dim_size[0] - 1){
    south = MPI_PROC_NULL;
  }else{
    MPI_Cart_rank(new_comm, neigh_coords, &south);
  }
  // Calculation of west neighbor
  neigh_coords[0] = coords[0];
  neigh_coords[1] = coords[1] - 1;
  if(neigh_coords[1] < 0){
    west = MPI_PROC_NULL;
  }else{
    MPI_Cart_rank(new_comm, neigh_coords, &west);
  }
  // Calculation of west neighbor
  neigh_coords[0] = coords[0];
  neigh_coords[1] = coords[1] + 1;
  if(neigh_coords[1] > dim_size[1] - 1){
    east = MPI_PROC_NULL;
  }else{
    MPI_Cart_rank(new_comm, neigh_coords, &east);
  }
  /************************/
  
  /* TEST FOR NEIGHBORS WITH 80 PROCESSES */
  if(my_world_rank == 45){ // FIRST TEST PASSED
    printf("Rank 45 has neighbors:\n");
    if(north == MPI_PROC_NULL){
      printf("ERROR - NORTH IS NULL\n");
    }else{
        MPI_Cart_coords(new_comm, north, 2, neigh_coords);
        printf("North: Rank %d == (%d, %d)\t", north, neigh_coords[0], neigh_coords[1]);
    }
    if(south == MPI_PROC_NULL){
      printf("ERROR - SOUTH IS NULL\n");
    }else{
        MPI_Cart_coords(new_comm, south, 2, neigh_coords);
        printf("South: Rank %d == (%d, %d)\t", south, neigh_coords[0], neigh_coords[1]);
    }
    if(east == MPI_PROC_NULL){
      printf("ERROR - EAST IS NULL\n");
    }else{
        MPI_Cart_coords(new_comm, east, 2, neigh_coords);
        printf("East: Rank %d == (%d, %d)\t", east, neigh_coords[0], neigh_coords[1]);
    }
    if(west == MPI_PROC_NULL){
      printf("ERROR - WEST IS NULL\n");
    }else{
        MPI_Cart_coords(new_comm, west, 2, neigh_coords);
        printf("West: Rank %d == (%d, %d)\t", west, neigh_coords[0], neigh_coords[1]);
    }
  }
  
  
  /* Cleaning up */
  free(u_local);
  u_local = NULL;
  free(u_old_local);
  u_old_local = NULL;
  free(fXsquared);
  fXsquared = NULL;
  free(fYsquared);
  fYsquared = NULL;
  MPI_Finalize();
  /***************/
	return 0;
}
