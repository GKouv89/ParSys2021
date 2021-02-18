#include <stdio.h>
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
  
  /* Local parameters calculation */
  double xRight = 1.0;
  double xLeft = -1.0;
  double yBottom = -1.0;
  double yUp = 1.0;
  double xLeft_local, xRight_local, yBottom_local, yUp_local, coords[2];
  int n_local, m_local;
  double deltaX_local, deltaY_local;
  double *u_local, *u_old_local, *f_local;

  if(comm_size != 80){
    double root, length;
    root = sqrt(param.n);
    length = xRight - xLeft;
    MPI_Cart_coords(new_comm, my_world_rank, 2, coords);
    xLeft_local = xLeft + ((double)coords[0])*(length/root);
    xRight_local = xLeft_local + (length/root);          
    yUp_local = yUp - ((double)coords[1])*(length/root);
    yBottom_local = yUp_local - (length/root);
    
    n_local = param.n/root;
    m_local = param.m/root;
    
    deltaX_local = (xLeft_local - xRight_local)/(n_local - 1);
    deltaY_local = (yUp_local - yBottom_local)/(m_local - 1);
    
    char *msg = malloc(512*sizeof(char));
    if(my_world_rank == 12){
      sprintf(msg, "Process rank 12 solves in [%.1lf,%.1lf]x[%.1lf,%.1lf] with n_local = %d and m_local = %d\n", xLeft_local, xRight_local, xLeft_local, xRight_local, n_local, m_local);
      MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, 0, new_comm);
    }else if(my_world_rank == 0){
      MPI_Recv(msg, 512, MPI_CHAR, 0, new_comm);
      printf("%s", msg);
      free(msg);
      msg = NULL;
    }

  }
  u_local = (double*)calloc((n_local + 2)*(m_local + 2), sizeof(double));
  u_old_local = (double*)calloc((n_local + 2)*(m_local + 2), sizeof(double));
  f = (double*)calloc(n_local*m_local, sizeof(double));
  
  if(u_local == NULL || u_old_local == NULL || f == NULL){
    if(my_world_rank == 0){
      printf("Not enough memory for 2 %ix%i plus one %ix%i matrices.\n", n_local + 2, m_local + 2, n_local, m_local);
    }
    MPI_Finalize();
    exit(1);
  }
  
  /********************************/
  
  /* Cleaning up */
  free(u_local);
  u_local = NULL;
  free(u_old_local);
  u_old_local = NULL;
  free(f);
  f = NULL;
  MPI_Finalize();
  /***************/
	return 0;
}
