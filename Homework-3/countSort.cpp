#include <cstdlib>
#include <vector>
#include <string>
#include <thread>
#include <omp.h>
#include <stdio.h>

// 1. If we try to parallelize the for i loop (the outer loop),\
 which variables should be private and which should be shared? (5 points)
    //Answer: the variables j, and count should be private while \
    temp and 'a' should be shared. j and count would both provide \
    race conditions if they weren't unique to the thread.

// 2. If we consider the memcpy implementation not thread-safe,\
 how would you approach parallelizing this operation? (5 points)
    //Answer: create a memory safe version of the function. Using an OMP parallel for loop\
    to copy the elements from temp to a.
    
// 3. Write a C/C++ OpenMP program that includes a parallel implementation\
 of count_sort. 30 points)


    void count_sort_thread_safe( int a[], int n) {
        int i, j, count;
        int * temp = static_cast<int*>(malloc(n* sizeof ( int )));
                
        #pragma omp parallel for private(count, j) //prevent race condition on threads 
        for (i = 0; i < n; i++) {//loop as we were given, but with braces.
            count = 0;
            for (j = 0; j < n; j++){ 
                if (a[j] < a[i]){
                    count++;
                }
                else if(a[j] == a[i] && j < i){
                    count++;
                }
            }
            temp[count] = a[i];
        }

        //thread safe 'memcpy'
        #pragma omp parallel for //No calculated values, so no need for private variables
            for (int i_o = 0; i_o < n; i_o++) {
                a[i_o] = temp[i_o]; //copy the values (input) number of threads at a time.
            }
        free(temp);
    } /* count_sort */

int main(int argc, const char** argv) {
    if (argc != 3) {
        printf("Usage: %s <num_threads> <num_elements>\n", argv[0]);
        return 1;
    }
    //set the omp number of threads to use in this program.
    omp_set_num_threads(std::stoi(argv[1]));

    int num_elements = std::stoi(argv[2]);
    int * a = static_cast<int *>(std::malloc(num_elements * sizeof(int)));

    srand(100);
    //use omp, fill a with random numbers
    #pragma omp parallel for shared(a) //the array is seeded, but the different\
    threads will fill the array differently, but with the same numbers every time.
        for (int i = 0; i < num_elements; i++) { //start of for loop
            a[i] = rand() % num_elements + 1;    //fill the array with a thread counted value
        }
    

    printf("original: ");
    for (int i = 0; i < num_elements; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    count_sort_thread_safe(a, num_elements);

    printf("sorted: ");
    for (int i = 0; i < num_elements; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    return 0;
}


     