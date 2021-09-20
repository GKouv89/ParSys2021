#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

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
  double *tmp_local, *u_local, *u_old_local, *fXsquared, *fYsquared;

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
    n_local = 105; // We have decided we have divided x axis in 10 subspaces
    m_local = 83; // We have decided to divide y axis in 8 subspaces
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

  /***** A small test for xLeft_local, xRight_local, yUp_local, yBottom_local *****/
  // if(my_world_rank == 0){
  //   printf("xLeft_local: %lf\n", xLeft_local);
  //   printf("xRight_local: %lf\n", xRight_local);
  //   printf("yUp_local: %lf\n", yUp_local);
  //   printf("yBottom_local: %lf\n", yBottom_local);
  // }
  /***** A small test for xLeft_local, xRight_local, yUp_local, yBottom_local *****/

  /* Datatypes for array row and column sending and receiving */
  // We require one data type for sending a row and one for sending a column
  // First, for the row.
  // We account for halo elements and 'extract' only the elements that are 
  // native to the current process' data.
  // Create the datatype
  MPI_Datatype row_type;
  MPI_Type_contiguous(m_local, MPI_DOUBLE, &row_type); // Length of row == number of columns
  MPI_Type_commit(&row_type);
  MPI_Datatype column_type;
  MPI_Type_vector(n_local, 1, m_local+1, MPI_DOUBLE, &column_type); // Count == length of column == number of rows
                                                                  // Stride == length of row == number of columns
  MPI_Type_commit(&column_type);
  /************************************************************/
  
  double maxAcceptableError = param.tol;
  double error = 0.0;
  int maxIterationCount = param.mits;
  int iterationCount = 0;
  
  #define SRC(XX,YY) u_old_local[(YY)*(n_local+2)+(XX)]
  #define DST(XX,YY) u_local[(YY)*(n_local+2)+(XX)]

  clock_t start = clock(), diff;
  double t1, t2;
  MPI_Barrier(new_comm);
  t1 = MPI_Wtime();

  // Coefficients
  double cx = 1.0/(param.deltaX*param.deltaX);
  double cy = 1.0/(param.deltaY*param.deltaY);
  double cc = -2.0*cx-2.0*cy-param.alpha;

  int x, y;
  double fX, fY;
  double updateVal;
  double f;

  for (y = 1; y < (m_local+1); y++)
  {
      fY = yBottom_local + (y-1)*param.deltaY;
      fYsquared[y-1] = fY*fY;
      for (x = 1; x < (n_local+1); x++)
      {
          fX = xLeft_local + (x-1)*param.deltaX;
          fXsquared[x-1] = fX*fX;
          f = -param.alpha*(1.0-fXsquared[x-1])*(1.0-fYsquared[y-1]) - 2.0*(2.0-fXsquared[x-1]-fYsquared[y-1]);
          updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                          (SRC(x,y-1) + SRC(x,y+1))*cy +
                          SRC(x,y)*cc - f
                      )/cc;
          DST(x,y) = SRC(x,y) - param.relax*updateVal;
          error += updateVal*updateVal;
      }
  }
  error = sqrt(error)/(n_local*m_local); // SOS: DO WE NEED THE LOCAL OR GLOBAL ONES HERE??
  iterationCount++;
  // Swap the buffers
  tmp_local = u_old_local;
  u_old_local = u_local;
  u_local = tmp_local;
  // if(my_world_rank == 0){
  //   printf("Completed first rep, where fXsquared and fYsquared are calculated.\n");
  // }

  /* Iterate as long as it takes to meet the convergence criterion */
  while (iterationCount < maxIterationCount && error > maxAcceptableError)
  {    	
    /* COMMUNICATION OF HALO POINTS */
    MPI_Request reception_requests[4], send_requests[4];
    // We receive from our north neighbor its southmost row, 
    // and we store that in our northermost one, so row 0, column 1 is our starting point
    MPI_Irecv(&(SRC(1, 0)), 1, row_type, north, 0, new_comm, &reception_requests[0]); 
    // We receive from our south neighbor its northermost row, 
    // and we store that in our southermost one, so row n_local + 1, column 1 is our starting point
    MPI_Irecv(&(SRC(1, n_local + 1)), 1, row_type, south, 0, new_comm, &reception_requests[1]); 
    
    // We receive from our east neighbor its westernmost column, 
    // and we store that in our easternmost one, so row 1, column m_local + 1 is our starting point
    MPI_Irecv(&(SRC(m_local + 1, 1)), 1, column_type, east, 0, new_comm, &reception_requests[3]); 
    
    // We receive from our west neighbor its easternnmost column, 
    // and we store that in our westernmost one, so row 1, column 0 is our starting point
    MPI_Irecv(&(SRC(0, 1)), 1, column_type, west, 0, new_comm, &reception_requests[2]); 
    
    // if(my_world_rank == 0){
    //   printf("Done with Irecv commands.\n");
    // }
    // We send our second northest row to our north neighbor
    // so row 1, column 1 is our starting point
    MPI_Isend(&(SRC(1,1)), 1, row_type, north, 0, new_comm, &send_requests[0]);
    // We send our second southest row to our south neighbor
    // so row n_local, column 1 is our starting point
    MPI_Isend(&(SRC(1, n_local)), 1, row_type, south, 0, new_comm, &send_requests[1]);
    // We send our second easternmost row to our east neighbor
    // so row 1, column 1 is our starting point
    MPI_Isend(&(SRC(1,1)), 1, column_type, east, 0, new_comm, &send_requests[2]);
    // We send our second westernmost row to our south neighbor
    // so row 1, column m_local is our starting point
    MPI_Isend(&(SRC(m_local, 1)), 1, column_type, west, 0, new_comm, &send_requests[3]);
    // if(my_world_rank == 0){
    //   printf("Done with Isend commands.\n");
    // }
              
    // FOR NOW, WE WON'T DO ANYTHING IN BETWEEN: WE'LL JUST WAIT TO RECEIVE THE HALO POINTS
    // AND PROCEED IN NORMAL CALCULATION, JUST TO SEE IF THE RESULTS ARE OKAY.
    // THEN WE'LL MAKE THIS MORE EFFICCIENT.

    MPI_Waitall(4, reception_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(4, send_requests, MPI_STATUSES_IGNORE);

    // if(my_world_rank == 0){
    //   printf("Done with waitall commands.\n");
    // }
  
    /********************************/

    error = 0.0;
    for (y = 1; y < (m_local+1); y++)
    {
        for (x = 1; x < (n_local+1); x++)
        {
            f = -param.alpha*(1.0-fXsquared[x-1])*(1.0-fYsquared[y-1]) - 2.0*(2.0-fXsquared[x-1]-fYsquared[y-1]);
            updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                            (SRC(x,y-1) + SRC(x,y+1))*cy +
                            SRC(x,y)*cc - f
                        )/cc;
            DST(x,y) = SRC(x,y) - param.relax*updateVal;
            error += updateVal*updateVal;
        }
    }
    error = sqrt(error)/(n_local*m_local);
    iterationCount++;
    // Swap the buffers
    tmp_local = u_old_local;
    u_old_local = u_local;
    u_local = tmp_local;
    // if(my_world_rank == 0){
    //   printf("Done with rep.\n");
    // }
  }

  t2 = MPI_Wtime();
  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  if(my_world_rank == 0){
    printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1 ); 
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
  }
  // printf("Residual %g\n",error);
  
  /* Cleaning up custom datatypes */
  MPI_Type_free(&row_type);
  MPI_Type_free(&column_type);
  /********************************/
  
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
