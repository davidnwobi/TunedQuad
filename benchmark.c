#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <stdlib.h>

// Case 1: Function that uses va_list to handle variable arguments
int sum_varargs(int count, ...) {
    va_list args;
    int sum = 0;

    va_start(args, count);
    for (int i = 0; i < count; i++) {
        sum += va_arg(args, int);
    }
    va_end(args);

    return sum;
}

// Case 2: Function that uses an array pointer with heap allocation (including allocation overhead)
int sum_array_heap(int count) {
    int sum = 0;
    int *args = (int *)malloc(count * sizeof(int));

    if (!args) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }

    for (int i = 0; i < count; i++) {
        args[i] = i + 1;
        sum += args[i];
    }

    free(args);
    return sum;
}

// Case 3: Function that uses an array allocated on the stack
int sum_array_stack(const int count) {
    int sum = 0;
    int args[count];  // Stack allocation

    for (int i = 0; i < count; i++) {
        args[i] = i + 1;
        sum += args[i];
    }

    return sum;
}

// Case 4: Function that takes arguments directly
int sum_direct(int a, int b, int c, int d, int e) {
    return a + b + c + d + e;
}

// Benchmarking function
void benchmark(int iterations, const int count) {
    int result;
    clock_t start, end;

    // Benchmark sum_varargs
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result = sum_varargs(count, 1, 2, 3, 4, 5);
    }
    end = clock();
    double time_varargs = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("sum_varargs: %f seconds\n", time_varargs);

    // Benchmark sum_array_heap
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result = sum_array_heap(count);
    }
    end = clock();
    double time_array_heap = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("sum_array_heap (with allocation): %f seconds\n", time_array_heap);

    // Benchmark sum_array_stack
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result = sum_array_stack(count);
    }
    end = clock();
    double time_array_stack = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("sum_array_stack: %f seconds\n", time_array_stack);

    // Benchmark sum_direct
    start = clock();
    for (int i = 0; i < iterations; i++) {
        result = sum_direct(1, 2, 3, 4, 5);
    }
    end = clock();
    double time_direct = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("sum_direct: %f seconds\n", time_direct);
}

int main() {
    int iterations = 10000000;
    const int count = 5;

    benchmark(iterations, count);

    return 0;
}
