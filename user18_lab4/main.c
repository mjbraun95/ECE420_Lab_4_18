/*
    MPI Implementation of Lab 4
    mpicc main.c Lab4_IO.c -o main -lm
    mpirun -np 4 -f ~/hosts ./main
*/
#define LAB4_EXTEND
#include "Lab4_IO.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

#define MAX_ITER 10000
#define EPSILON 0.00001
#define DAMPING_FACTOR 0.85

int main(int argc, char* argv[]) {
    int world_rank, world_size, nodecount;
    struct node* nodehead;
    double* r, *r_pre, *r_global;
    int i, j, local_n_count;
    double start, end;
    int* recvcounts, *displs;
    int iteration_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Debug: Confirm start of each process
    // printf("Process %d started\n", world_rank);
    fflush(stdout);

    // Broadcast the node count from the master process to all other processes
    if (world_rank == 0) {
        FILE* ip;
        if ((ip = fopen("data_input_meta", "r")) == NULL) {
            // printf("Error opening the data_input_meta file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fscanf(ip, "%d", &nodecount);
        fclose(ip);
    }
    
    MPI_Bcast(&nodecount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // printf("Process %d received nodecount %d\n", world_rank, nodecount);
    fflush(stdout);

    if (node_init(&nodehead, 0, nodecount) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    r = (double*)malloc(nodecount * sizeof(double));
    r_pre = (double*)malloc(nodecount * sizeof(double));
    r_global = (double*)malloc(nodecount * sizeof(double));
    recvcounts = (int*)malloc(world_size * sizeof(int));
    displs = (int*)malloc(world_size * sizeof(int));

    for (i = 0; i < nodecount; ++i) {
        r[i] = 1.0 / nodecount;
    }

    int chunk = nodecount / world_size;
    int remainder = nodecount % world_size;
    int local_start = world_rank * chunk + (world_rank < remainder ? world_rank : remainder);
    int local_end = local_start + chunk + (world_rank < remainder);
    local_n_count = local_end - local_start;

    // printf("Process %d handling nodes from %d to %d\n", world_rank, local_start, local_end - 1);
    fflush(stdout);

    MPI_Allgather(&local_n_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    displs[0] = 0;
    for (i = 1; i < world_size; ++i) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    GET_TIME(start);
    do {
        vec_cp(r, r_pre, nodecount);

        for (i = local_start; i < local_end; ++i) {
            double sum = 0;
            for (j = 0; j < nodehead[i].num_in_links; ++j) {
                int inlink_node_index = nodehead[i].inlinks[j];
                sum += r_pre[inlink_node_index] / nodehead[inlink_node_index].num_out_links;
            }
            r[i] = (1.0 - DAMPING_FACTOR) / nodecount + DAMPING_FACTOR * sum;
        }

        MPI_Allgatherv(r + local_start, local_n_count, MPI_DOUBLE,
                       r_global, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // Debug: Check the integrity of r_global in all processes
        if (iteration_count == 1) {
            // printf("Process %d, Sample r_global[0]: %f, r_global[%d]: %f\n", world_rank, r_global[0], nodecount-1, r_global[nodecount-1]);
            fflush(stdout);
        }

        vec_cp(r_global, r, nodecount);
        // printf("Process %d, Iteration %d, Relative Error: %f\n", world_rank, iteration_count, rel_error(r, r_pre, nodecount));
        fflush(stdout);

        iteration_count++;
    } while (rel_error(r, r_pre, nodecount) >= EPSILON && iteration_count < MAX_ITER);
    GET_TIME(end);

    if (world_rank == 0) {
        Lab4_saveoutput(r, nodecount, end - start);
    }

    free(r);
    free(r_pre);
    free(r_global);
    free(recvcounts);
    free(displs);
    node_destroy(nodehead, nodecount); // Freeing the nodehead after use
    MPI_Finalize();
    return 0;
}
