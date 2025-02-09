#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>

// #define DEBUG

#ifdef DEBUG
    #define log(FMT, ...) printf(FMT, ##__VA_ARGS__)
    int DEBUG_c; //could be a char. Used to pause the program
    #define bp scanf("%c", &DEBUG_c)
#else
    #define bp
    #define log(...)
#endif // DEBUG

using namespace std;


void globalSumThread(int threadID, vector<float>& data, vector<float>& bin_maxes, 
    vector<vector<int>>& separatedBinCounts, int num_data, int num_threads, int num_bins){
    // vector<int> localBinCounts(num_bins);

    for(int i = threadID; i < num_data; i+=num_threads){
        int low = 0;
        int high = num_bins - 1;
        int mid;
        while(low <= high){
            mid = low+(high-low)/2;
            if(mid == 0 && data[i] <= bin_maxes[0]) {
                separatedBinCounts[threadID][0]++;
                break;
            }
            else if(data[i] <= bin_maxes[mid] && (mid == 0 || data[i] > bin_maxes[mid-1])) {
                separatedBinCounts[threadID][mid]++;
                break;
            } else if(data[i] > bin_maxes[mid]) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }

    for(int i = 0; i < separatedBinCounts[threadID].size(); i++){
        log("ThreadID: %d, bin %d has %d \n", threadID, i, separatedBinCounts[threadID][i]);
    }
    return;
}


int main(int argc, char *argv[])
{
    // <num_threads>, the number of threads to use for the execution
    // <num_bins>, the number of bins in the histogram
    // <min_meas>, minimum (float) value of the measurements
    // <max_meas>, maximum (float) value of the measurements
    // <num_data>, number of measurements

    //Check the number of arguments
    if (argc != 6)    {
        std::cerr << "Usage: " << argv[0] << " <number of threads> <bin count> <min meas> <max meas> <data count>" << std::endl;
        return 1;
    }
    //evaluate the executability of the arguments
    int num_threads = std::stoi(argv[1]);
    vector<thread> threads;

    int   num_bins = stoi(argv[2]);
    if (num_bins <= 0) {
        std::cerr << "Error: num_bins must be greater than 0" << std::endl;
        return 1;
    }
    float min_meas  = stof(argv[3]);
    float max_meas  = stof(argv[4]);
    if(min_meas < 0 || max_meas < 0){
        std::cerr << "Error: min_meas and max_meas must be greater than 0" << std::endl;
        return 1;
    }
    if (min_meas >= max_meas)    {
        std::cerr << "Error: min_meas must be less than max_meas" << std::endl;
        return 1;
    }

    int num_data  = stoi(argv[5]);
        // bin_maxes: a list containing the upper bound of each bin
        // bin_counts: a list containing the number of elements in each bin
    vector<float> bin_maxes(num_bins);
    vector<int> bin_counts(num_bins); //total
    vector<vector<int>> separatedBinCounts(num_threads, vector<int>(num_bins)); //
    // populate an array (data) of <num_data> float elements between <min_meas>\
     and <max_meas>. Use srand(100) to initialize your pseudorandom sequence.
    vector<float> data(num_data);
    srand(100);
    for (int i = 0; i < num_data; i++) {
        float temp = min_meas + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_meas - min_meas)));
        data[i] = temp;
    }

    // get the maximum values and store them into the bin_maxes vector
        //The first element will be used to calculate all the rest. use of the max_element and min_element functions from the algorithm library to 
        //promote efficiency
    float bin_width = (*max_element(data.begin(), data.end()) - *min_element(data.begin(), data.end())) / num_bins;
    float min_val = *min_element(data.begin(), data.end());

    bin_maxes[0] = min_val + bin_width;
    for (int i = 1; i < num_bins-1; i++) {
        bin_maxes[i] = bin_maxes[i-1] + bin_width;
    }
    //can't be calculated, so a quick grab of the max element keeps the algorithm working.
    bin_maxes[num_bins-1] = *max_element(data.begin(), data.end());

    // compute the histogram (i.e., bin_maxes and num_bins) using\
      <number of threads> threads using a global sum
    for(int i = 0; i < num_threads; i++){
        // threads.push_back(thread(globalSumThread, i, data, bin_maxes, 
        //                         bin_counts, num_data, num_threads, num_bins));
        threads.push_back(thread(globalSumThread, i, ref(data), ref(bin_maxes), ref(separatedBinCounts), 
                                num_data, num_threads, num_bins));
    }

    // globalSumThread(data, bin_maxes, bin_counts, num_threads, 0);

    // compute the histogram (i.e., bin_maxes and num_bins) using\
      <number of threads> threads using a tree structured sum
    for (auto& thread : threads) {
        thread.join();
    }
    //after barrier, compute all threads in global manner
    for(int i = 0; i < num_threads; i++){
        for(int j = 0; j < num_bins; j++){
            bin_counts[j] += separatedBinCounts[i][j];
        }
    }

    for(int i = 0; i < num_bins; i++){
        printf("bin[%d] : %d \n", i, bin_counts[i]);
    }

    // ./histogram 4 10 0.0 5.0 100
    // The outputs of the program should be the same for both implementations\
     (global sum and tree structured sum):
    // Your program should print all list elements on a single line and print\
     each list on its own line. Additionally, label each list something like \
     the following example output.

    return 0;
}




//Original instructions:

// Write (and upload) a program using Pthreads or C++11 threads that implements\
 the histogram program discussed in "4 - ParallelSoftware". 

// The program will have to:

// populate an array (data) of <num_data> float elements between <min_meas>\
 and <max_meas>. Use srand(100) to initialize your pseudorandom sequence.
// compute the histogram (i.e., bin_maxes and num_bins) using  <number of threads>\
 threads using a global sum
// compute the histogram (i.e., bin_maxes and num_bins) using  <number of threads>\
 threads using a tree structured sum
// The inputs of the program are:

// <number of threads>, the number of threads to use for the execution
// <num_bins>, the number of bins in the histogram
// <min_meas>, minimum (float) value of the measurements
// <max_meas>, maximum (float) value of the measurements
// <num_data>, number of measurements
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
