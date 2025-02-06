#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <thread>

using namespace std;


void globalSumThread(int threadID){

    // this_thread::sleep_for(chrono::milliseconds(rand() % 1000));
    // printf("Hello from thread : %d\n", threadID);

    return;
}


int main(int argc, char *argv[])
{
    // <number of threads>, the number of threads to use for the execution
    // <bin_count>, the number of bins in the histogram
    // <min_meas>, minimum (float) value of the measurements
    // <max_meas>, maximum (float) value of the measurements
    // <data_count>, number of measurements

    //Check the number of arguments
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <number of threads> <bin count> <min meas> <max meas> <data count>" << std::endl;
        return 1;
    }
    //evaluate the executability of the arguments
    int thread_count = std::stoi(argv[1]);
    vector<thread> threads;

    int bin_count   = stoi(argv[2]);
    float min_meas  = stof(argv[3]);
    float max_meas  = stof(argv[4]);

    if (min_meas >= max_meas)
    {
        std::cerr << "Error: min_meas must be less than max_meas" << std::endl;
        return 1;
    }

    int data_count  = stoi(argv[5]);
        // bin_maxes: a list containing the upper bound of each bin
        // bin_counts: a list containing the number of elements in each bin
    vector<float> bin_maxes(bin_count);
    vector<int> bin_counts(bin_count);

    // populate an array (data) of <data_count> float elements between <min_meas>\
     and <max_meas>. Use srand(100) to initialize your pseudorandom sequence.

    vector<float> data(data_count);
    srand(100);
    for (int i = 0; i < data_count; ++i) {
        data[i] = min_meas + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_meas - min_meas)));
    }

    // for (int i = 0; i < data_count; ++i) {
    //     cout << data[i] << "\t";
    //     if (i % 10 == 0) {
    //         cout << endl;
    //     }
    // }
    // cout << endl;


    // compute the histogram (i.e., bin_maxes and bin_count) using\
      <number of threads> threads using a global sum
    for(int i = 0; i < thread_count; i++){
        threads.push_back(thread(globalSumThread, i));
    }
    float globalSum;
    // globalSumThread(data, bin_maxes, bin_counts, thread_count, 0);

    // compute the histogram (i.e., bin_maxes and bin_count) using\
      <number of threads> threads using a tree structured sum
    for (auto& thread : threads) {
        thread.join();
    }



    
    // ./histogram 4 10 0.0 5.0 100
    // The outputs of the program should be the same for both implementations (global sum and tree structured sum):
    // Your program should print all list elements on a single line and print each list on its own line. Additionally, label each list something like the following example output.

    return 0;
}




//Original instructions:

// Write (and upload) a program using Pthreads or C++11 threads that implements\
 the histogram program discussed in "4 - ParallelSoftware". 

// The program will have to:

// populate an array (data) of <data_count> float elements between <min_meas>\
 and <max_meas>. Use srand(100) to initialize your pseudorandom sequence.
// compute the histogram (i.e., bin_maxes and bin_count) using  <number of threads>\
 threads using a global sum
// compute the histogram (i.e., bin_maxes and bin_count) using  <number of threads>\
 threads using a tree structured sum
// The inputs of the program are:

// <number of threads>, the number of threads to use for the execution
// <bin_count>, the number of bins in the histogram
// <min_meas>, minimum (float) value of the measurements
// <max_meas>, maximum (float) value of the measurements
// <data_count>, number of measurements
// Your program must adhere to the following order of command-line arguments:

// <number of threads> <bin count> <min meas> <max meas> <data count>
// That is, if you name your executable 'histogram' an example command line is:

// ./histogram 4 10 0.0 5.0 100
// The outputs of the program should be the same for both implementations (global sum\
 and tree structured sum):

// bin_maxes: a list containing the upper bound of each bin
// bin_counts: a list containing the number of elements in each bin
// Your program should print all list elements on a single line and print each list on\
 its own line. Additionally, label each list something like the following example output.

// Global Sum
// bin_maxes = 0.500 1.000 1.500 2.000 2.500 3.000 3.500 4.000 4.500 5.000 
// bin_counts = 7 8 10 10 8 10 12 9 12 14 
// Tree Structured Sum
// bin_maxes = 0.500 1.000 1.500 2.000 2.500 3.000 3.500 4.000 4.500 5.000 
// bin_counts = 7 8 10 10 8 10 12 9 12 14 
// Important: always comment your code and explain what each function and code fragment\
 is doing. A poorly documented code will not get full score.

// Important: note that for the same input and varying number of threads the output must\
 be the same. Discuss the reasons why results might differ for different executions (or\
  number of threads) and what technique did you implemented to solve this problem.

 

// For CS6030 ONLY (+10 points)

// Time the execution of your program for varying number of threads from 1 to 8 and\
 produce a plot.
// Notes:
// - time only the computation of the histogram (i.e., not the creation/population/\
destruction of the input array)
// - use a large enough input array to show strong scaling of your program
// - you can put the plot and the program into a zip file and attach this as answer\
 to this question.
