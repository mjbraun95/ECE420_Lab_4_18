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
    int i, j;
    double start, end;
    int* recvcounts, *displs;
    int iteration_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Load graph and broadcast nodecount
    if (world_rank == 0) {
        if ((node_init(&nodehead, 0, nodecount)) != 0) {
            printf("Failed to initialize nodes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&nodecount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocation for PageRank vectors
    r = (double*)malloc(nodecount * sizeof(double));
    r_pre = (double*)malloc(nodecount * sizeof(double));
    r_global = (double*)malloc(nodecount * sizeof(double));
    recvcounts = (int*)malloc(world_size * sizeof(int));
    displs = (int*)malloc(world_size * sizeof(int));

    // Initialize PageRank values
    for (i = 0; i < nodecount; ++i) {
        r[i] = 1.0 / nodecount;
    }

    // Determine local work range
    int chunk = nodecount / world_size;
    int remainder = nodecount % world_size;
    int local_start = world_rank * chunk + (world_rank < remainder ? world_rank : remainder);
    int local_end = local_start + chunk + (world_rank < remainder);

    // Collect information on the number of elements each process will send
    int local_n_count = local_end - local_start;
    MPI_Allgather(&local_n_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements
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

        // Gather all partial PageRank vectors to all processes using MPI_Allgatherv
        MPI_Allgatherv(r + local_start, local_n_count, MPI_DOUBLE,
                       r_global, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // Copy global results back to local r for next iteration
        vec_cp(r_global, r, nodecount);

        iteration_count++;
    } while (rel_error(r, r_pre, nodecount) >= EPSILON && iteration_count < MAX_ITER);
    GET_TIME(end);

    // Save output and clean up in the master process
    if (world_rank == 0) {
        Lab4_saveoutput(r, nodecount, end - start);
    }

    free(r);
    free(r_pre);
    free(r_global);
    free(recvcounts);
    free(displs);
    MPI_Finalize();
    return 0;
}