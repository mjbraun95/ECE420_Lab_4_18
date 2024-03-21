#define LAB4_EXTEND // Ensure struct node and related functions are visible
#include <mpi.h>
#include "Lab4_IO.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

#define MAX_ITER 10000
#define EPSILON 0.00001
#define DAMPING_FACTOR 0.85

// Function to perform vector copy
void vec_cp(double *vec_src, double *vec_dst, int size) {
    for (int i = 0; i < size; ++i) {
        vec_dst[i] = vec_src[i];
    }
}

int main(int argc, char* argv[]){
    int world_rank, world_size, nodecount;
    struct node *nodehead;
    double *r, *r_pre, *r_global;
    int i, j, local_start, local_end;
    double start, end;
    FILE *ip;
    int iteration_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // The first step is to load the graph and get the nodecount
    // This step should be done before calling node_init()
    // Assuming Lab4_load_graph() is a function to load the graph and return the nodecount
    if (world_rank == 0) {
        if ((ip = fopen("data_input_meta", "r")) == NULL){
            printf("Error opening the data_input_meta file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fscanf(ip, "%d", &nodecount);
        fclose(ip);
    }
    
    // Broadcast the nodecount to all processes
    MPI_Bcast(&nodecount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Now we can safely initialize nodehead with the correct nodecount
    if (node_init(&nodehead, 0, nodecount)) MPI_Abort(MPI_COMM_WORLD, 1);

    // Allocation for PageRank vectors after nodecount is correctly initialized
    r = (double*)malloc(nodecount * sizeof(double));
    r_pre = (double*)malloc(nodecount * sizeof(double));
    r_global = (double*)malloc(nodecount * sizeof(double));

    // Initialize PageRank values
    for (i = 0; i < nodecount; ++i) r[i] = 1.0 / nodecount;

    // Allocate memory for inverse_num_out_links array
    double* inverse_num_out_links = (double*)malloc(nodecount * sizeof(double));
    if (inverse_num_out_links == NULL) {
        printf("Error: Memory allocation failed for inverse_num_out_links.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Compute inverse_num_out_links values
    for (int i = 0; i < nodecount; ++i) {
        inverse_num_out_links[i] = 1.0 / nodehead[i].num_out_links;
    }

    // Determine local work range
    local_start = world_rank * (nodecount / world_size);
    local_end = (world_rank + 1) * (nodecount / world_size);
    if (world_rank == world_size - 1) local_end = nodecount; // Last process

    GET_TIME(start);
    do {
        vec_cp(r, r_pre, nodecount);

        for (i = local_start; i < local_end; ++i) {
            double sum = 0;
            for (j = 0; j < nodehead[i].num_in_links; ++j) {
                int inlink_node_index = nodehead[i].inlinks[j];
                sum += r_pre[inlink_node_index] * inverse_num_out_links[inlink_node_index];
            }
            r[i] = (1 - DAMPING_FACTOR) / nodecount + DAMPING_FACTOR * sum;
        }

        // Gather all partial PageRank vectors to all processes
        MPI_Allgather(r + local_start, local_end - local_start, MPI_DOUBLE,
                      r_global, local_end - local_start, MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Copy global results back to local r for next iteration
        vec_cp(r_global, r, nodecount);

        iteration_count++;
    } while(rel_error(r, r_pre, nodecount) >= EPSILON && iteration_count < MAX_ITER);
    GET_TIME(end);

    // Save output and clean up in the master process
    if (world_rank == 0) {
        Lab4_saveoutput(r, nodecount, end - start);
    }

    free(r); free(r_pre); free(r_global);
    free(inverse_num_out_links);
    MPI_Finalize();
    return 0;
}
