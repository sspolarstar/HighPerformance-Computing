
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <vector>
#include "semaphore.h"

const int MAX_THREADS = 1024;
const int MSG_MAX = 100;

/* Global variables:  accessible to all threads */
int thread_count;
char** messages;

std::vector<Semaphore>* sems;

void Usage(char* prog_name);
void *Send_msg(void* rank);  /* Thread function */

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   long       thread;
   
   if (argc != 2) Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);
   if (thread_count <= 0 || thread_count > MAX_THREADS) Usage(argv[0]);

   std::vector<std::thread> thread_handles(thread_count);

   messages = (char**)malloc(thread_count*sizeof(char*));
   
   sems = new std::vector<Semaphore>(thread_count);

   for (thread = 0; thread < thread_count; thread++) {
      messages[thread] = NULL;
      sems[thread].wait(thread);
   }

   for (thread = 0; thread < thread_count; thread++)
      thread_handles[thread] = thread(Send_msg, thread);

   for (thread = 0; thread < thread_count; thread++) {
      thread_handles[thread].join();
   }

   for (thread = 0; thread < thread_count; thread++) {
      free(messages[thread]);
   }
   free(messages);

   return 0;
}  /* main */


/*--------------------------------------------------------------------
 * Function:    Usage
 * Purpose:     Print command line for function and terminate
 * In arg:      prog_name
 */
void Usage(char* prog_name) {

   fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
   exit(0);
}  /* Usage */


/*-------------------------------------------------------------------
 * Function:       Send_msg
 * Purpose:        Create a message and ``send'' it by copying it
 *                 into the global messages array.  Receive a message
 *                 and print it.
 * In arg:         rank
 * Global in:      thread_count
 * Global in/out:  messages, semaphores
 * Return val:     Ignored
 * Note:           The my_msg buffer is freed in main
 */
void *Send_msg(void* rank) {
   long my_rank = (long) rank;
   long dest = (my_rank + 1) % thread_count;
   char* my_msg = (char*) malloc(MSG_MAX*sizeof(char));

   sprintf(my_msg, "Hello to %ld from %ld", dest, my_rank);
   messages[dest] = my_msg;
   sems[dest].notify(dest);  /* "Unlock" the semaphore of dest */

   sems[my_rank].wait(my_rank);  /* Wait for our semaphore to be unlocked */
   printf("Thread %ld > %s\n", my_rank, messages[my_rank]);

   return NULL;
}  /* Send_msg */
