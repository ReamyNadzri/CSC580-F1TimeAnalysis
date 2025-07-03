#include <stdio.h>
#include <omp.h>
#include <float.h>
#include <windows.h>
#include <psapi.h>

// Helper function to find min/max in an array
void find_min_max(double arr[], int n, double *min_val, double *max_val)
{
    *min_val = arr[0];
    *max_val = arr[0];
    for (int i = 1; i < n; i++)
    {
        if (arr[i] < *min_val)
            *min_val = arr[i];
        if (arr[i] > *max_val)
            *max_val = arr[i];
    }
}

// function to get the memory usage
size_t get_peak_memory_kb()
{
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.PeakWorkingSetSize / 1024; // Convert bytes to KB
}

int main()
{
    double start_time = omp_get_wtime(); // Start timing

    // 3 Datasets
    double driver_a_laps[] = {85.34, 84.95, 85.11, 92.34, 84.88, 85.05, 84.99, 86.12, 90.50, 85.21};
    double driver_b_laps[] = {84.75, 84.91, 85.88, 84.82, 91.60, 85.15, 84.79, 85.33, 89.98, 85.01};
    double driver_c_laps[] = {86.10, 85.55, 85.43, 84.92, 88.88, 85.67, 93.10, 85.29, 86.04, 85.77};

    int n_a = sizeof(driver_a_laps) / sizeof(driver_a_laps[0]);
    int n_b = sizeof(driver_b_laps) / sizeof(driver_b_laps[0]);
    int n_c = sizeof(driver_c_laps) / sizeof(driver_c_laps[0]);

    // Variables to hold the result from each section/thread
    double min_a, max_a;
    double min_b, max_b;
    double min_c, max_c;

    printf("--- F1 Lap Time Analysis using OpenMP Sections ---\n");
    printf("\nProcessing each driver's data in a separate parallel section...\n");

#pragma omp parallel sections
    {
#pragma omp section
        {
            // Thread 1 processes Driver A's data
            find_min_max(driver_a_laps, n_a, &min_a, &max_a);
            printf("Thread %d processed Driver A\n", omp_get_thread_num());
        }

#pragma omp section
        {
            // Thread 2 processes Driver B's data
            find_min_max(driver_b_laps, n_b, &min_b, &max_b);
            printf("Thread %d processed Driver B\n", omp_get_thread_num());
        }

#pragma omp section
        {
            // Thread 3 processes Driver C's data
            find_min_max(driver_c_laps, n_c, &min_c, &max_c);
            printf("Thread %d processed Driver C\n", omp_get_thread_num());
        }
    } // End of parallel sections. All threads sync here.

    // --- Final Reduction Step (Sequential) ---
    // Now find the overall min and max from the results of each section.
    double fastest_lap = min_a;
    if (min_b < fastest_lap)
        fastest_lap = min_b;
    if (min_c < fastest_lap)
        fastest_lap = min_c;

    double slowest_lap = max_a;
    if (max_b > slowest_lap)
        slowest_lap = max_b;
    if (max_c > slowest_lap)
        slowest_lap = max_c;

    double end_time = omp_get_wtime();         // Stop timing
    size_t peak_mem_kb = get_peak_memory_kb(); // Measure memory

    printf("\n==================================================\n");
    printf("       F1 Lap Time Analysis using OpenMP\n");
    printf("==================================================\n");
    printf("Overall Fastest Lap Time (min): %.2f seconds\n", fastest_lap);
    printf("Overall Slowest Lap Time (max): %.2f seconds\n", slowest_lap);

    // Consistent output style with MPI version
    printf("Total Execution Time          : %f seconds\n", end_time - start_time);
    printf("Peak Memory Usage             : %zu KB\n", peak_mem_kb);
    printf("==================================================\n");

    return 0;
}
