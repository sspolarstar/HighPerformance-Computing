#include <cstdlib>
#include <vector>
#include <string>
#include <thread>
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <stdio.h>

void count_sort( int a[], int n) {
    int i, j, count;
    int * temp = static_cast<int*>(malloc(n* sizeof ( int )));
    
    for (i = 0; i < n; i++) {
        count = 0;
        for (j = 0; j < n; j++)
        if (a[j] < a[i]){
            count++;
        }
        else if(a[j] == a[i] && j < i){
            count++;
        }
        temp[count] = a[i];
    }
    memcpy(a, temp, n*sizeof(int)); //NOT THREAD SAFE
    free(temp);
  } /* count_sort */

int main(int argc, const char** argv) {
    if argc != 3 {
        printf("Usage: %s <num_threads> <num_elements>\n", argv[0]);
        return 1;
    }
    int num_threads = std::stoi(argv[1]);
    int num_elements = std::stoi(argv[2]);
    int * a = (int *) malloc(num_elements * sizeof(int));

    // Generate random integers in the range 1 to n (initialize with srand(100) ) 
    // print the original sequence
    // print the ordered sequence
    // Your program should print all list elements on a single line and print each\
     list on its own line. Additionally, label each list something like the following \
     example output.
    srand(100);
    for (int i = 0; i < num_elements; i++) {
        a[i] = rand() % num_elements + 1;
    }

    printf("original: ");
    for (int i = 0; i < num_elements; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    return 0;
}

// 1. If we try to parallelize the for i loop (the outer loop),\
 which variables should be private and which should be shared? (5 points)
    //Answer: the variables i, j, and count should be private while \
    temp and 'a' should be shared

// 2. If we consider the memcpy implementation not thread-safe,\
 how would you approach parallelizing this operation? (5 points)
    //create a memory safe version of the function. Using a OMP critical section\
     to ensure that only one thread can access the memory at a time will also\
     allow additional safety.
    
// 3. Write a C/C++ OpenMP program that includes a parallel implementation\
 of count_sort. 30 points)
     