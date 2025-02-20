#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <condition_variable>
#include <mutex>

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


//////////////////////CODE Obtained by using AI ////////////////////////
class Barrier {
public:
    Barrier(unsigned int count) : threshold(count), count(count), generation(0) {}
    
    void wait() {
        std::unique_lock<std::mutex> lock(m);
        unsigned int gen = generation;
        if (--count == 0) {
            generation++;
            count = threshold; // Reset for next use
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return generation != gen; });
        }
    }
    
private:
    std::mutex m;
    std::condition_variable cv;
    const unsigned int threshold;
    unsigned int count;
    unsigned int generation;
};
/////////////////////////END OF AI CODE///////////////////////////////



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


void treeSumThread(int threadID, vector<float>& data, vector<float>& bin_maxes, vector<vector<int>>& separatedBinCounts, 
                   int num_data, int num_threads, int num_bins, Barrier& barrier) {
    // 1. Local Histogram Computation
    for (int i = threadID; i < num_data; i += num_threads) {
        int low = 0, high = num_bins - 1, mid;
        while (low <= high) {
            mid = low + (high - low) / 2;
            if (mid == 0 && data[i] <= bin_maxes[0]) {
                separatedBinCounts[threadID][0]++;
                break;
            }
            else if (data[i] <= bin_maxes[mid] && (mid == 0 || data[i] > bin_maxes[mid - 1])) {
                separatedBinCounts[threadID][mid]++;
                break;
            } 
            else if (data[i] > bin_maxes[mid]) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }

    // Barrier: Wait for all threads to finish the local histogram
    barrier.wait();

    // 2. Tree Reduction Phase
    // For log2(num_threads) steps, merge partner thread’s results into the current thread
    for (int step = 1; step < num_threads; step *= 2) {
        if (threadID % (2 * step) == 0) {
            int partner = threadID + step;
            if (partner < num_threads) {
                for (int bin = 0; bin < num_bins; bin++) {
                    separatedBinCounts[threadID][bin] += separatedBinCounts[partner][bin];
                }
            }
        }
        // Barrier: Wait for all threads to finish this reduction step before moving on
        barrier.wait();
    }
}



int main(int argc, char *argv[])
{

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

    auto start_time_global = std::chrono::high_resolution_clock::now();

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

    auto end_time_global = std::chrono::high_resolution_clock::now();


    printf("Global Sum\n");
    printf("bin_maxes = ");
    for(int i = 0; i < num_bins; i++){
        printf("%f ", bin_maxes[i]);
    }
    printf("\n");
    printf("bin counts = ");
    for(int i = 0; i < num_bins; i++){
        printf("%d ", bin_counts[i]);
    }
    printf("\n");



    bin_counts.clear();
    bin_counts.resize(num_bins);

    separatedBinCounts.clear();
    separatedBinCounts.resize(num_threads, vector<int>(num_bins));

    threads.clear();

    //introduce the barrier for mutex control.
    Barrier barrier(num_threads);

    auto start_time_tree = std::chrono::high_resolution_clock::now();


    threads.clear();
    for (int i = 0; i < num_threads; i++) { 
        threads.push_back(thread(treeSumThread, i, ref(data), ref(bin_maxes),
                                ref(separatedBinCounts), num_data, num_threads, num_bins, ref(barrier)));
    }

    // Wait for all threads to complete
    for(auto& thread : threads) {
        thread.join();
    }

    // Results are now in separatedBinCounts[0]
    bin_counts = separatedBinCounts[0];

    auto end_time_tree = std::chrono::high_resolution_clock::now();

    
    // ./histogram.o 4 10 0.0 5.0 100

    // Your program should print all list elements on a single line and print\
     each list on its own line. Additionally, label each list something like \
     the following example output.

    printf("Tree Structured Sum\n");
    printf("bin_maxes = ");
    for(int i = 0; i < num_bins; i++){
        printf("%f ", bin_maxes[i]);
    }
    printf("\n");
    printf("bin counts = ");
    for(int i = 0; i < num_bins; i++){
        printf("%d ", bin_counts[i]);
    }
    printf("\n");

    std::chrono::duration<double> elapsed = end_time_global - start_time_global;
    std::chrono::duration<double> elapsed_tree = end_time_tree - start_time_tree;

    printf("Time taken for global sum: %f seconds\n", elapsed.count()*10);
    printf("Time taken for tree sum: %f seconds\n", elapsed_tree.count()*10);
    return 0;
}


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
