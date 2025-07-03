#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <windows.h>
#include <psapi.h>

/*
 ============================================================================
 CSC580: Parallel Processing Project
 Case Study: F1 Race Performance Analysis using MPI
 ============================================================================
*/

// function to get the memory usage
size_t get_peak_memory_kb() {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.PeakWorkingSetSize / 1024; // bytes to KB
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Initialize MPI

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);  // Get total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Get the rank ID of current process

    const int TOTAL_LAPS = 30;
    double* all_laps = NULL;  // This will store all lap times in rank 0
    double start_time, end_time, elapsed_time;

    // Only rank 0 will initialize the lap data
    if (rank == 0) {
        all_laps = (double*)malloc(TOTAL_LAPS * sizeof(double));

        // 3 Datasets
        double driver_a_laps[] = {85.34, 84.95, 85.11, 92.34, 84.88, 85.05, 84.99, 86.12, 90.50, 85.21};
        double driver_b_laps[] = {84.75, 84.91, 85.88, 84.82, 91.60, 85.15, 84.79, 85.33, 89.98, 85.01};
        double driver_c_laps[] = {86.10, 85.55, 85.43, 84.92, 88.88, 85.67, 93.10, 85.29, 86.04, 85.77};

        // Combine all lap data into one big array
        for (int i = 0; i < 10; i++) {
            all_laps[i]      = driver_a_laps[i];
            all_laps[i + 10] = driver_b_laps[i];
            all_laps[i + 20] = driver_c_laps[i];
        }

        start_time = MPI_Wtime(); // Start timing only on rank 0
    }

    // Divide total laps among processes
    int laps_per_proc = TOTAL_LAPS / num_procs;

    // Each process allocates memory for its portion
    double* sub_laps = (double*)malloc(laps_per_proc * sizeof(double));

    // Scatter data from rank 0 to all processes
    MPI_Scatter(all_laps, laps_per_proc, MPI_DOUBLE, sub_laps, laps_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each process calculates its own local min and max
    double local_min = DBL_MAX;
    double local_max = -DBL_MAX;
    for (int i = 0; i < laps_per_proc; i++) {
        if (sub_laps[i] < local_min) local_min = sub_laps[i];
        if (sub_laps[i] > local_max) local_max = sub_laps[i];
    }

    // Use MPI_Reduce to find global min and max lap times at rank 0
    double global_min, global_max;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // arrays to gather all local mins and maxs 
    double* gathered_mins = NULL;
    double* gathered_maxs = NULL;
    if (rank == 0) {
        gathered_mins = (double*)malloc(num_procs * sizeof(double));
        gathered_maxs = (double*)malloc(num_procs * sizeof(double));
    }

    // Gather all local mins and maxs from each process to rank 0
    MPI_Gather(&local_min, 1, MPI_DOUBLE, gathered_mins, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_max, 1, MPI_DOUBLE, gathered_maxs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Only rank 0 prints the final results
    if (rank == 0) {
        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;
        size_t peak_mem_kb = get_peak_memory_kb();

        // Display summary of results
        printf("\n==================================================\n");
        printf("       F1 Lap Time Analysis using MPI\n");
        printf("==================================================\n");
        printf("Overall Fastest Lap Time (Min): %.2f seconds\n", global_min);
        printf("Overall Slowest Lap Time (Max): %.2f seconds\n", global_max);
        printf("Total Execution Time          : %f seconds\n", elapsed_time);
        printf("Peak Memory Usage             : %zu KB\n", peak_mem_kb);  
        printf("==================================================\n");

        // Free memory on rank 0
        free(all_laps);
        free(gathered_mins);
        free(gathered_maxs);
    }

    // Free local memory
    free(sub_laps);

    MPI_Finalize();  
    return 0;
}
