/*
    Serial Implementation of Lab 4
    gcc main_template.c Lab4_IO.c -o main_template -lm

*/

#define LAB4_EXTEND

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Lab4_IO.h"
#include "timer.h"

#define EPSILON 0.00001
#define DAMPING_FACTOR 0.85

int main (int argc, char* argv[]){
    // instantiate variables
    struct node *nodehead;
    int nodecount;
    double *r, *r_pre;
    int i, j;
    int iterationcount;
    double start, end;
    FILE *ip;
    /* INSTANTIATE MORE VARIABLES IF NECESSARY */

    // load data 
    if ((ip = fopen("data_input_meta","r")) == NULL) {
        printf("Error opening the data_input_meta file.\n");
        return 253;
    }
    fscanf(ip, "%d\n", &nodecount);
    fclose(ip);
    if (node_init(&nodehead, 0, nodecount)) return 254;
    
    // initialize variables
    r = malloc(nodecount * sizeof(double));
    r_pre = malloc(nodecount * sizeof(double));
    
    GET_TIME(start);
    iterationcount = 0;
    for ( i = 0; i < nodecount; ++i)
        r[i] = 1.0 / nodecount;
    /* INITIALIZE MORE VARIABLES IF NECESSARY */

    // Core calculation
    do {
        vec_cp(r, r_pre, nodecount); // Copy current PageRank scores to r_pre

        for (i = 0; i < nodecount; ++i) {
            double sum = 0;
            for (j = 0; j < nodehead[i].num_in_links; ++j) {
                int inlink_node_index = nodehead[i].inlinks[j];
                sum += r_pre[inlink_node_index] / nodehead[inlink_node_index].num_out_links;
            }
            r[i] = (1 - DAMPING_FACTOR) / nodecount + DAMPING_FACTOR * sum;
        }

        ++iterationcount;
    } while (rel_error(r, r_pre, nodecount) >= EPSILON);
    GET_TIME(end);

    Lab4_saveoutput(r, nodecount, end - start);

    // post processing
    node_destroy(nodehead, nodecount);
    free(r); free(r_pre);
    return 0;
}
