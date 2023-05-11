#include <stdlib.h>
#include <stdio.h>
#include "/usr/include/openmpi-x86_64/mpi.h"
#include "board.h"

#define N 20
#define MAXITERS 200

//TODO CHANGE TO COLUMN WISE
// 1. make a MPI_Datatype_vector for a column (glej primer v mpi_datatypes.c)
// 2. use MPI_Type_commit to commit the datatype
// 3. make the "extended" version for the new datatype so that it can be used for communication
// 4. use the new datatype in MPI_Scatter and MPI_Sendrecv
// 5. fix the computation of the neighbours
// 6. fix the gathering of the results

int main(int argc, char* argv[])
{
	int i, j, neighs;
	int iters = 0;

	char * boardptr = NULL;					// ptr to board
	char ** board;							// board, 2D matrix, contignous memory allocation!

	int procs, myid;			
	int mystart, myend, myrows;
	char ** myboard;						// part of board that belongs to a process
	char ** myboard_new;					// myboard_new is of the same size as myboard, needed for correct computation of iteration steps
	char * myrow_top, * myrow_bot;			// data (row) from top neighbour, data (row) from bottom neighbour

	MPI_Init(&argc, &argv);					// initiailzation
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);	// process ID
	MPI_Comm_size(MPI_COMM_WORLD, &procs);	// number of processes

	// initialize global board
	if (myid == 0)
	{
		srand(1573949136);
		board = board_initialize(N, N);
		boardptr = *board;
		board_print(board, N, N);
	}
	// divide work
	mystart = N / procs * myid;				// determine scope of work for each process; process 0 also works on its own part
	myend = N / procs * (myid + 1);
	myrows = N / procs;

	// initialize my structures
	myboard = board_initialize(myrows, N);
	myboard_new = board_initialize(myrows, N);
	myrow_top = (char*)malloc(N * sizeof(char));
	myrow_bot = (char*)malloc(N * sizeof(char));

	// scatter initial matrix
	MPI_Scatter(boardptr, myrows * N, MPI_CHAR, 
				*myboard, myrows * N, MPI_CHAR, 
				0, MPI_COMM_WORLD);
	// ptr to data (NULL on receiving processes), size of data sent to each process, data type, 
	// ptr to process data, size of received data, received data type, 
	// sender, communicator

	// do the calculation
	while (iters < MAXITERS)
	{
		// exchange borders with neigbouring processes
		MPI_Sendrecv(myboard[0], N, MPI_CHAR, (myid + procs - 1) % procs, 0,
					 myrow_bot, N, MPI_CHAR, (myid + 1) % procs, 0,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		// ptr to send data, send data size, send data type, receiver, message tag,
		// ptr to received data, received data size, recevied data type, sender, message tag,
		// communicator, status
		MPI_Sendrecv(myboard[myrows - 1], N, MPI_CHAR, (myid + 1) % procs, 1,
					 myrow_top, N, MPI_CHAR, (myid + procs - 1) % procs, 1, 
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		// do the computation of my part
		for (i = 0; i < myrows; i++)
			for (j = 0; j < N; j++)
			{
				neighs = count_neighbours_mpi(myboard, myrow_top, myrow_bot, myrows, N, i, j);
				if (neighs == 3 || (myboard[i][j] == 1 && neighs == 2))
					myboard_new[i][j] = 1;
				else
					myboard_new[i][j] = 0;
			}
		iters++;
		// swap boards (iter --> iter + 1)
		board_update(&myboard, &myboard_new);
	}
	
	// gather results
	MPI_Gather(*myboard, myrows * N, MPI_CHAR, 
			   boardptr, myrows * N, MPI_CHAR, 
			   0, MPI_COMM_WORLD);
	// data to send, send data size, data type,
	// gathered data, recevied data size, data type,
	// gathering process, communicator

	// display
	if (myid == 0)
		board_print(board, N, N);

	// free memory
	if (myid == 0)
		board_free(board);
	board_free(myboard);
	free(myrow_top);
	free(myrow_bot);

	MPI_Finalize();			// finalize MPI

	return 0;
}
