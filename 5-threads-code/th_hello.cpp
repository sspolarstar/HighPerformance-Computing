/* File:  
 *    cth_hello.cpp
 * Purpose:
 *    Illustrate basic use of C++11 threads:  create some threads,
 *    each of which prints a message.
 * Input:
 *    none
 * Output:
 *    message from each thread
 * Compile:  mpic++ th_hello.cpp
 * Usage:    mpirun a.out uses all available cores
 *           mpirun --use-hwthread-cpus a.out uses all available hardware threads
 *           mpirun -np 4 a.out uses 4 cores
 *           mpirun --use-hwthread-cpus -np 8 a.out uses 8 hardware threads
 *           mpirun -np 1 a.out
 *           ./a.out
 * References:
 *    IPP:   Section 4.2 (p. 153)
 *    https://stackoverflow.com/questions/52272399/the-advantage-of-c11-threads
 *    https://stackoverflow.com/questions/1452721/why-is-using-namespace-std-considered-bad-practice
 *    https://larryullman.com/forums/index.php?/topic/2567-getting-rid-of-those-std-pre-designators/
 *    https://thispointer.com/c11-how-to-get-a-thread-id/
 */
 
#include <iostream>
#include <vector>
#include <thread>

using namespace std;
 
const int MAX_THREADS = 64;

/* Global variable:  accessible to all threads */
int thread_count;  

void Usage(char* prog_name);
void Hello(long rank); /* Thread function */


int main(int argc, char* argv[]) {
   // parse args
   if (argc != 2) Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);  
   if (thread_count <= 0 || thread_count > MAX_THREADS) Usage(argv[0]);

   vector<thread> thread_handles(thread_count); 

   // spawn the threads
   long t;  /* Use long in case of a 64-bit system */
   for (long t = 0; t<thread_count; t++) 
     thread_handles[t] = thread(Hello, t);

   printf("Hello from the main thread\n");

   // wait for them to finish by calling join
   for(t=0; t<thread_count; t++) 
      thread_handles[t].join();

   return 0;
}

void Hello(long my_rank) {
   printf("Hello from thread %ld of %d\n", my_rank, thread_count);
}

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
   fprintf(stderr, "0 < number of threads <= %d\n", MAX_THREADS);
   exit(0);
}  /* Usage */
