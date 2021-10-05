#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <omp.h>

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

double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
            error += localError*localError;
        }
    }
    return error; // We return this so process with rank zero
    // will aggregate these and calculate overall error.
    // The problem is: will this be correct in respects to maxXcount -2 and maxYCount - 2
    // adding up to overall maxXcount and maxYcount?
}

int main(int argc, char* argv[]){
  int prov;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);
  /* Cartesian Grid Creation */
	int comm_size, my_world_rank, dim_size[2], periods[2];
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	int dimsA[2] = {0, 0};
  MPI_Dims_create(comm_size, 2, dimsA);
  // if(my_world_rank == 0){
  //   printf("Dimensions by MPI_Dims_create for %d processes: (%d, %d).\n", comm_size, dimsA[0], dimsA[1]);
  // }
  switch(comm_size){
    case 4:
      dim_size[1] = dim_size[0] = 2;
      break;
    default:
      dim_size[0] = dimsA[0];
      dim_size[1] = dimsA[1];
      break;
  }
	periods[0] = periods[1] = 0;
	MPI_Comm new_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_size, periods, 1, &new_comm);
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

    printf("-> %d, %d, %g, %g, %g, %d\n", param.n, param.m, param.alpha, param.relax, param.tol, param.mits);
    printf("comm_size = %d\tdimsA[0] = %d\tdimsA[1] = %d\n", comm_size, dimsA[0], dimsA[1]);
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
  double root /*, length*/;

  MPI_Cart_coords(new_comm, my_world_rank, 2, coords);
  switch(comm_size){
    default:
      n_local = param.n/dimsA[1]; 
      m_local = param.m/dimsA[0]; 
      xLeft_local = xLeft + ((double)coords[1])*n_local*param.deltaX;
      yBottom_local = yBottom + ((double)coords[0])*m_local*param.deltaY; 
      break;
    case 4:
      n_local = param.n/2;
      m_local = param.m/2;
      xLeft_local = xLeft + ((double)coords[1])*n_local*param.deltaX;
      yBottom_local = yBottom +((double)coords[0])*m_local*param.deltaY;
      break;
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
  int north, south, east, west;
  MPI_Cart_shift(new_comm, 1, 1, &west, &east);
  MPI_Cart_shift(new_comm, 0, 1, &north, &south);
  /************************/

  /* Datatypes for array row and column sending and receiving */
  // We require one data type for sending a row and one for sending a column
  // First, for the row.
  // We account for halo elements and 'extract' only the elements that are 
  // native to the current process' data.
  // Create the datatype
  MPI_Datatype row_type;
  MPI_Type_contiguous(n_local, MPI_DOUBLE, &row_type); // Length of row == number of columns
  MPI_Type_commit(&row_type);
  MPI_Datatype column_type;
  MPI_Type_vector(m_local, 1, n_local + 2, MPI_DOUBLE, &column_type); // Count == length of column == number of rows
                                                                     // Stride == length of row == number of columns
  MPI_Type_commit(&column_type);
  /************************************************************/
  
  double maxAcceptableError = param.tol;
  double error = 15;
  int maxIterationCount = param.mits;
  int iterationCount = 0;
  
  #define SRC(XX,YY) u_old_local[(YY)*(n_local+2)+(XX)]
  #define DST(XX,YY) u_local[(YY)*(n_local+2)+(XX)]

  // Coefficients
  double cx = 1.0/(param.deltaX*param.deltaX);
  double cy = 1.0/(param.deltaY*param.deltaY);
  double cc = -2.0*cx-2.0*cy-param.alpha;

  int x, y;
  double fX, fY;
  double updateVal;
  double f;
  
  MPI_Request *send_requests_current = (MPI_Request *)malloc(4*sizeof(MPI_Request));
  MPI_Request *send_requests_former = (MPI_Request *)malloc(4*sizeof(MPI_Request));
  MPI_Request *send_requests_temp;


  MPI_Request *receive_requests_current = (MPI_Request *)malloc(4*sizeof(MPI_Request));
  MPI_Request *receive_requests_former = (MPI_Request *)malloc(4*sizeof(MPI_Request));
  MPI_Request *receive_requests_temp;

  MPI_Barrier(new_comm);
  clock_t start = clock(), diff;
  double t1, t2;
  t1 = MPI_Wtime(); 
  MPI_Pcontrol(1);

  // We send our second northest row to our north neighbor
  // so row 1, column 1 is our starting point
  MPI_Send_init(&(SRC(1,1)), 1, row_type, north, 0, new_comm, &send_requests_current[0]);
  MPI_Send_init(&(DST(1,1)), 1, row_type, north, 0, new_comm, &send_requests_former[0]);
  
  // We send our second southest row to our south neighbor
  // so row n_local, column 1 is our starting point
  MPI_Send_init(&(SRC(1, m_local)), 1, row_type, south, 0, new_comm, &send_requests_current[1]);
  MPI_Send_init(&(DST(1, m_local)), 1, row_type, south, 0, new_comm, &send_requests_former[1]);
    
  // We send our second easternmost row to our east neighbor
  // so row 1, column 1 is our starting point
  MPI_Send_init(&(SRC(1,1)), 1, column_type, west, 0, new_comm, &send_requests_current[2]);
  MPI_Send_init(&(DST(1,1)), 1, column_type, west, 0, new_comm, &send_requests_former[2]);
    
  // We send our second westernmost row to our south neighbor
  // so row 1, column m_local is our starting point
  MPI_Send_init(&(SRC(n_local, 1)), 1, column_type, east, 0, new_comm, &send_requests_current[3]);
  MPI_Send_init(&(DST(n_local, 1)), 1, column_type, east, 0, new_comm, &send_requests_former[3]);

  // We receive from our north neighbor its southmost row, 
  // and we store that in our northermost one, so row 0, column 1 is our starting point
  MPI_Recv_init(&(SRC(1, 0)), 1, row_type, north, 0, new_comm, &receive_requests_current[0]); 
  MPI_Recv_init(&(DST(1, 0)), 1, row_type, north, 0, new_comm, &receive_requests_former[0]); 
  // We receive from our south neighbor its northermost row, 
  // and we store that in our southermost one, so row n_local + 1, column 1 is our starting point
  MPI_Recv_init(&(SRC(1, m_local + 1)), 1, row_type, south, 0, new_comm, &receive_requests_current[1]); 
  MPI_Recv_init(&(DST(1, m_local + 1)), 1, row_type, south, 0, new_comm, &receive_requests_former[1]); 
  // We receive from our east neighbor its westernmost column, 
  // and we store that in our easternmost one, so row 1, column m_local + 1 is our starting point
  MPI_Recv_init(&(SRC(n_local + 1, 1)), 1, column_type, east, 0, new_comm, &receive_requests_current[2]); 
  MPI_Recv_init(&(DST(n_local + 1, 1)), 1, column_type, east, 0, new_comm, &receive_requests_former[2]); 
  // We receive from our west neighbor its easternnmost column, 
  // and we store that in our westernmost one, so row 1, column 0 is our starting point
  MPI_Recv_init(&(SRC(0, 1)), 1, column_type, west, 0, new_comm, &receive_requests_current[3]); 
  MPI_Recv_init(&(DST(0, 1)), 1, column_type, west, 0, new_comm, &receive_requests_former[3]); 

  #pragma omp parallel 
  {
    #pragma omp for private(fY) schedule(static)
    for (y = 1; y < (m_local+1); y++)
    {
        fY = yBottom_local + (y-1)*param.deltaY;
        fYsquared[y-1] = fY*fY;
    }
    #pragma omp for private(fX) schedule(static)
    for (x = 1; x < (n_local+1); x++)
    {
        fX = xLeft_local + (x-1)*param.deltaX;
        fXsquared[x-1] = fX*fX;
    }

    while (iterationCount < maxIterationCount && error > maxAcceptableError)
    {    	
      #pragma omp barrier
      #pragma omp master
      {
        /* COMMUNICATION OF HALO POINTS */
        MPI_Startall(4, receive_requests_current);
        MPI_Startall(4, send_requests_current);
        
        error = 0.0;        
      }
      #pragma omp barrier
  
      #pragma omp for private(f, updateVal, x) reduction(+:error) schedule(static)  /* collapse(2) */
      for (y = 2; y < m_local; y++)
      {      
        for (x = 2; x < n_local; x++)
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
      // #pragma omp atomic
      //   error += partial_error;

      #pragma omp barrier
      #pragma omp master
      {
        MPI_Waitall(4, receive_requests_current, MPI_STATUSES_IGNORE);
      }
      #pragma omp barrier

      // Columns and rows that need halo 

      // Halo is:
        // x: 2nd and second-to-last (number 1 & n_local)
        // y: 2nd and second-to-last (number 1 & m_local)
      #pragma omp single
      y = 1;
      #pragma omp for reduction(+:error) private(f, updateVal) schedule(static)
      for (x = 1; x < n_local + 1; x++)
      {
          f = -param.alpha*(1.0-fXsquared[x-1])*(1.0-fYsquared[y-1]) - 2.0*(2.0-fXsquared[x-1]-fYsquared[y-1]);
          updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                          (SRC(x,y-1) + SRC(x,y+1))*cy +
                          SRC(x,y)*cc - f
                      )/cc;
          DST(x,y) = SRC(x,y) - param.relax*updateVal;
          error += updateVal*updateVal;
      }
      #pragma omp single
      y = m_local;      
      #pragma omp for reduction(+:error) private(f, updateVal) schedule(static)
      for (x = 1; x < n_local + 1; x++)
      {
          f = -param.alpha*(1.0-fXsquared[x-1])*(1.0-fYsquared[y-1]) - 2.0*(2.0-fXsquared[x-1]-fYsquared[y-1]);
          updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                          (SRC(x,y-1) + SRC(x,y+1))*cy +
                          SRC(x,y)*cc - f
                      )/cc;
          DST(x,y) = SRC(x,y) - param.relax*updateVal;
          error += updateVal*updateVal;
      }
      #pragma omp single
      x = 1;
      #pragma omp for reduction(+:error) private(f, updateVal) schedule(static)
      for (y = 2; y < m_local; y++)
      {
        f = -param.alpha*(1.0-fXsquared[x-1])*(1.0-fYsquared[y-1]) - 2.0*(2.0-fXsquared[x-1]-fYsquared[y-1]);
        updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                        (SRC(x,y-1) + SRC(x,y+1))*cy +
                        SRC(x,y)*cc - f
                    )/cc;
        DST(x,y) = SRC(x,y) - param.relax*updateVal;
        error += updateVal*updateVal;
      }
      #pragma omp single
      x = n_local;
      #pragma omp for reduction(+:error) private(f, updateVal) schedule(static)
      for (y = 2; y < m_local; y++)
      {
          f = -param.alpha*(1.0-fXsquared[x-1])*(1.0-fYsquared[y-1]) - 2.0*(2.0-fXsquared[x-1]-fYsquared[y-1]);
          updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                          (SRC(x,y-1) + SRC(x,y+1))*cy +
                          SRC(x,y)*cc - f
                      )/cc;
          DST(x,y) = SRC(x,y) - param.relax*updateVal;
          error += updateVal*updateVal;
      }

      #pragma omp barrier
      #pragma omp master
      {
        MPI_Allreduce(&error, &error, 1, MPI_DOUBLE, MPI_SUM, new_comm);
        error = sqrt(error)/(param.n*param.m);

        iterationCount++;
        // Swap the buffers
        tmp_local = u_old_local;
        u_old_local = u_local;
        u_local = tmp_local;
        MPI_Waitall(4, send_requests_current, MPI_STATUSES_IGNORE);

        send_requests_temp = send_requests_former;
        send_requests_former = send_requests_current;
        send_requests_current = send_requests_temp;

        receive_requests_temp = receive_requests_former;
        receive_requests_former = receive_requests_current;
        receive_requests_current = receive_requests_temp;
      }
      #pragma omp barrier
      /********************************/
    }
  }

  /* Iterate as long as it takes to meet the convergence criterion */

  MPI_Pcontrol(0);
  t2 = MPI_Wtime();
  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  
  if(my_world_rank == 0){
    printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1 ); 
    // printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    printf("Residual %g\n", error);
  }
  
  /* Cleaning up custom datatypes */
  MPI_Type_free(&row_type);
  MPI_Type_free(&column_type);
  /********************************/

  // u_old holds the solution after the most recent buffers swap
  double absoluteError = checkSolution(xLeft_local, yBottom_local,
                                        n_local+2, m_local+2,
                                        u_old_local,
                                        param.deltaX, param.deltaY,
                                        param.alpha);
  int my_cart_rank;
  double absoluteErrorSum;
  MPI_Comm_rank(new_comm, &my_cart_rank);
  MPI_Reduce(&absoluteError, &absoluteErrorSum, 1, MPI_DOUBLE, MPI_SUM, 0, new_comm);
  if(my_cart_rank == 0){
    absoluteErrorSum = sqrt(absoluteErrorSum)/(param.n*param.m);
    printf("The error of the iterative solution is %g\n", absoluteErrorSum);
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
  free(receive_requests_current);
  free(receive_requests_former);
  free(send_requests_current);
  free(send_requests_former);
  MPI_Finalize();
  /***************/
	return 0;
}
