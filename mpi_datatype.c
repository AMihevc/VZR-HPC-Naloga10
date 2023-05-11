// module load OpenMPI/4.1.0-GCC-10.2.0 
// mpicc mpi_datatype.c -o mpi_datatype
// srun --reservation=fri --mpi=pmix -n2 -N2 ./mpi_datatype

//////////////////////////////////////////////////////////////
// Compute product v x M on multiple MPI processes.
// Each process computes one dot product.
// Uses derived MPI data types.
/////////////////////////////////////////////////////////////
#include <stdio.h>
#include <string.h> 
#include <stdlib.h>
#include <mpi.h>

// define size of the matrix (N x M) and vector (N)
// number of columns in the matrix (M) is equal to the number of processes
#define N 10
#define M num_p

// function to compute the dot product
double dot(double  * v, double * c, int n){
    double result=0.0;
    for (int i=0;i<n;i++)
        result+=v[i]*c[i];
    return result;
}

int main(int argc, char *argv[]) 
{ 
	int rank; // process rank 
	int num_p; // total number of processes 
	int source; // sender rank
	int destination; // receiver rank 
	int tag = 0; // message tag 
	char message[100]; // message buffer 
	MPI_Status status; // message status 
    MPI_Datatype row, column, column_resized;

	MPI_Init(&argc, &argv); // initialize MPI 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get process rank 
	MPI_Comm_size(MPI_COMM_WORLD, &num_p); // get number of processes
   
    // create new data type for row vector
    MPI_Type_contiguous(N,MPI_DOUBLE,&row);
    MPI_Type_commit(&row);

    // create new data type for column vector
    MPI_Type_vector(N,1,M,MPI_DOUBLE,&column);
    MPI_Type_commit(&column);
    
    // resize the colmn vector data type to correctly scatter the columns of the matrix
    MPI_Type_create_resized(column, 0, sizeof(double), &column_resized);
    MPI_Type_commit(&column_resized);

    // print size info about new data types
    if (rank == 0)
    {
        MPI_Aint lb, extent;
        int s;
        MPI_Type_get_extent(row, &lb, &extent);
        MPI_Type_size(row, &s);
        printf("\nrow: lower bound: %d upper bound: %d size: %ld\n", lb, lb+extent, s);
        MPI_Type_get_extent(column, &lb, &extent);
        MPI_Type_size(column, &s);
        printf("column: lower bound: %d upper bound: %d size: %ld\n", lb, lb+extent, s);
        MPI_Type_get_extent(column_resized, &lb, &extent);
        MPI_Type_size(column_resized, &s);
        printf("column_resized: lower bound: %d upper bound: %d size: %ld\n\n", lb, lb+extent, s);
    }

    // allocate memory and initialize data
    double *Mat, *vec_result;
    double *vec = (double *)malloc(N*sizeof(double));
    double *col_vec = (double *)malloc(N*sizeof(double));
	if( rank == 0 ) 
	{ 
        Mat = (double *)malloc(N*M*sizeof(double));
        vec_result = (double *)malloc(M*sizeof(double));
        for(int r=0;r<N;r++){
            vec[r]=1.0;
            for(int c=0;c<M;c++){
                Mat[r*M+c]=1;
            }
        }
        fflush(stdout);
	}
    // distribute vector and the matrix to the processes
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(vec,1,row,0,MPI_COMM_WORLD);
    MPI_Scatter(Mat,1,column_resized,col_vec,1,row,0,MPI_COMM_WORLD);

    // compute dot product on each matrix column
    double result=dot(vec,col_vec,N);
    // gather the results and print
    MPI_Gather(&result,1,MPI_DOUBLE,vec_result,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0){
        printf("Result:[ ");
        for(int i=0;i<M;i++)
            printf("%.1f ", vec_result[i]);
        printf("]\n");    
    }

    // free new data types and memory
    if(rank==0){
        free(vec_result);
        free(Mat);
    }
    free(vec);
    free(col_vec);

    MPI_Type_free(&row);
    MPI_Type_free(&column);
    MPI_Type_free(&column_resized);
    MPI_Finalize();

   
    
	return 0; 
} 
