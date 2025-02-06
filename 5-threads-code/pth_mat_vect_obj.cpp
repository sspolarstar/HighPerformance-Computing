

#include <iostream>
//#include <stdlib.h>
#include <thread>

/* Global variables */
int     thread_count;
int     m, n;
double* A;
double* x;
double* y;

/* Serial functions */
void Usage(char* prog_name);
void Read_matrix(char* prompt, double A[], int m, int n);
void Read_vector(char* prompt, double x[], int n);
void Print_matrix(char* title, double A[], int m, int n);
void Print_vector(char* title, double y[], double m);
void Populate_matrix(double A[], int m, int n);
void Populate_vector(double y[], int m);

class MatVec{
public:
   MatVec(long rank){
      my_rank = rank;
   }

   void operator()(){
      std::thread::id id = std::this_thread::get_id();
      std::cout << "running thread rank " << my_rank << " thread id " << id << std::endl;

      int i, j;
      int local_m = m/thread_count; 
      int my_first_row = my_rank*local_m;
      int my_last_row = (my_rank+1)*local_m - 1;

      for (i = my_first_row; i <= my_last_row; i++) {
         y[i] = 0.0;
         for (j = 0; j < n; j++)
             y[i] += A[i*n+j]*x[j];
      }
   }
private:
   long my_rank;
};

/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   long       thread;

   //if (argc != 2) Usage(argv[0]);
   //thread_count = atoi(argv[1]);
   thread_count = 4;
   std::thread thread_handles[thread_count];

   //printf("Enter m and n\n");
   //scanf("%d%d", &m, &n);
   m=4;
   n=3;

   A = new double[m*n];
   x = new double[n];
   y = new double[m];
   
   //Read_matrix("Enter the matrix", A, m, n);
   Populate_matrix(A, m, n);
   Print_matrix("Matrix", A, m, n);

   Populate_vector(x, n);
   Print_vector("Vector", x, n);

   for (thread = 0; thread < thread_count; thread++){
   	thread_handles[thread] = std::thread(MatVec(thread));
      std::cout << "created thread param " << thread << " thread id " << thread_handles[thread].get_id() << std::endl;
   }

   for (thread = 0; thread < thread_count; thread++)
      thread_handles[thread].join();

   Print_vector("The product is", y, m);

   delete [] A;
   delete [] x;
   delete [] y;

   return 0;
}  /* main */


/*------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   print a message showing what the command line should
 *            be, and terminate
 * In arg :   prog_name
 */
void Usage (char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count>\n", prog_name);
   exit(0);
}  /* Usage */

/*------------------------------------------------------------------
 * Function:    Read_matrix
 * Purpose:     Read in the matrix
 * In args:     prompt, m, n
 * Out arg:     A
 */
void Read_matrix(char* prompt, double A[], int m, int n) {
   int             i, j;

   printf("%s\n", prompt);
   for (i = 0; i < m; i++) 
      for (j = 0; j < n; j++)
         scanf("%lf", &A[i*n+j]);
}  /* Read_matrix */

void Populate_matrix(double A[], int m, int n) {
   int             i, j;
   for (i = 0; i < m; i++) 
      for (j = 0; j < n; j++)
         A[i*n+j] = rand()%100;
}  

/*------------------------------------------------------------------
 * Function:        Read_vector
 * Purpose:         Read in the vector x
 * In arg:          prompt, n
 * Out arg:         x
 */
void Read_vector(char* prompt, double x[], int n) {
   int   i;

   printf("%s\n", prompt);
   for (i = 0; i < n; i++) 
      scanf("%lf", &x[i]);
}  /* Read_vector */

void Populate_vector(double x[], int n) {
   int   i;

   for (i = 0; i < n; i++) 
      x[i]=rand()%100;
}  


/*------------------------------------------------------------------
 * Function:    Print_matrix
 * Purpose:     Print the matrix
 * In args:     title, A, m, n
 */
void Print_matrix( char* title, double A[], int m, int n) {
   int   i, j;

   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%4.1f ", A[i*n + j]);
      printf("\n");
   }
}  /* Print_matrix */


/*------------------------------------------------------------------
 * Function:    Print_vector
 * Purpose:     Print a vector
 * In args:     title, y, m
 */
void Print_vector(char* title, double y[], double m) {
   int   i;

   printf("%s\n", title);
   for (i = 0; i < m; i++)
      printf("%4.1f ", y[i]);
   printf("\n");
}  /* Print_vector */
