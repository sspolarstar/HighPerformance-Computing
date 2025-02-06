/* File:     pth_pi_mutex.c
 * Purpose:  Estimate pi using series 
 *
 *              pi = 4*[1 - 1/3 + 1/5 - 1/7 + 1/9 - . . . ]
 *
 *           This version uses a mutex to protect the critical section
 *
 * Compile:  g++ -g -Wall -o th_pi_mutex th_pi_mutex.cpp -lm -lpthread
 *           timer.h needs to be available 
 * Run:      ./pth_pi_mutex <number of threads> <n>
 *           n is the number of terms of the Maclaurin series to use
 *           n should be evenly divisible by the number of threads
 * 
 * Input:    none            
 * Output:   The estimate of pi using multiple threads, one thread, and the 
 *           value computed by the math library arctan function
 *           Also elapsed times for the multithreaded and singlethreaded
 *           computations.
 *
 * Notes:
 *    1.  The radius of convergence for the series is only 1.  So the 
 *        series converges quite slowly.
 *
 * IPP:   Section 4.6 (pp. 168 and ff.)
 */        

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include "timer.h"

using namespace std;
const int MAX_THREADS = 1024;

long thread_count;
long long n;

/* parallel functions */
void Thread_sum(double &sum, mutex &mtx, int rank);

/* Only executed by main thread */
void Get_args(int argc, char* argv[]);
void Usage(char* prog_name);
double Serial_pi(long long n);

int main(int argc, char* argv[]) {
   long t;  /* Use long in case of a 64-bit system */
   double start, finish, elapsed;

   /* Get number of threads from command line */
   Get_args(argc, argv);

   vector<thread> thread_handles(thread_count);

   double sum = 0.0; 
   mutex mtx;

   GET_TIME(start);
   for (t = 0; t < thread_count; t++) 
      thread_handles[t] = thread(Thread_sum, ref(sum), ref(mtx), t);

   for (t = 0; t < thread_count; t++) 
      thread_handles[t].join();

   GET_TIME(finish);
   elapsed = finish - start;

   sum = 4.0*sum;
   printf("With n = %lld terms,\n", n);
   printf("   Our estimate of pi = %.15f\n", sum);
   printf("The elapsed time is %e seconds\n", elapsed);
   GET_TIME(start);
   sum = Serial_pi(n);
   GET_TIME(finish);
   elapsed = finish - start;
   printf("   Single thread est  = %.15f\n", sum);
   printf("The elapsed time is %e seconds\n", elapsed);
   printf("                   pi = %.15f\n", 4.0*atan(1.0));

   return 0;
   }  /* main */

/*------------------------------------------------------------------
 * Function:       Thread_sum
 * Purpose:        Add in the terms computed by the thread running this 
 * In arg:         rank
 * Return val:     ignored
 * Globals in:     n, thread_count
 * Global in/out:  sum 
 */
void Thread_sum(double &sum, mutex &mtx, int rank) {
   double factor, my_sum = 0.0;
   int my_n = n/thread_count;
   int my_first_i = my_n*rank;
   int my_last_i  = my_first_i + my_n;

   if (my_first_i % 2 == 0) 
      factor = 1.0;
   else 
      factor = -1.0;

   for (int i=my_first_i; i<my_last_i; i++, factor = -factor) {
      my_sum += factor/(2*i+1);  
   }
   
   mtx.lock();
   sum += my_sum;  
   mtx.unlock();
   }  /* Thread_sum */

/*------------------------------------------------------------------
 * Function:   Serial_pi
 * Purpose:    Estimate pi using 1 thread
 * In arg:     n
 * Return val: Estimate of pi using n terms of Maclaurin series
 */
double Serial_pi(long long n) {
   double sum = 0.0;
   long long i;
   double factor = 1.0;

   for (i = 0; i < n; i++, factor = -factor) {
      sum += factor/(2*i+1);
   }
   return 4.0*sum;

}  /* Serial_pi */

/*------------------------------------------------------------------
 * Function:    Get_args
 * Purpose:     Get the command line args
 * In args:     argc, argv
 * Globals out: thread_count, n
 */
void Get_args(int argc, char* argv[]) {
   if (argc != 3) Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);  
   if (thread_count <= 0 || thread_count > MAX_THREADS) Usage(argv[0]);
   n = strtoll(argv[2], NULL, 10);
   if (n <= 0) Usage(argv[0]);
}  /* Get_args */


/*------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Print a message explaining how to run the program
 * In arg:    prog_name
 */
void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <number of threads> <n>\n", prog_name);
   fprintf(stderr, "   n is the number of terms and should be >= 1\n");
   fprintf(stderr, "   n should be evenly divisible by the number of threads\n");
   exit(0);
}  /* Usage */
