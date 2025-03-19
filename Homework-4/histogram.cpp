// compute a histogram:
// this is the input line <bin count> <min meas> <max meas> <data_count>

//The input could look like this:
//mpiexec -n 2 ./histogram 10 0.0 5.0 100 
//it must use mpi. Specifically a reduction summation.

//The output will again be a list of counts and maxes:
// bin_maxes: 0.500 1.000 1.500 2.000 2.500 3.000 3.500 4.000 4.500 5.000 
// bin_counts: 7 8 10 10 8 10 12 9 12 14

// Rules:
// In your MPI implementation:

// have process 0 read in the inputs data and distribute them among all the processes
// have process 0 populate an array (data) of <data_count> float elements between <min_meas> and <max_meas>. Use srand(100) to initialize your pseudorandom sequence.
// have process 0 distribute portions of the pseudorandom sequence to the other processors (note: do not share the entire array with all other processes)
// compute the histogram 
// have process 0 print out the outputs (i.e., bin_maxes and bin_counts)
// Verify that for different numbers of processes the result is the same


//verification:
/*
scott@cmp:Homework-4$ mpic++ -g -Wall -o mpi_histogram histogram.cpp
scott@cmp:Homework-4$ mpiexec -n 7 ./mpi_histogram 10 0.0 5.0 100
bin_maxes: 0.588489 1.07443 1.56036 2.0463 2.53224 3.01818 3.50412 3.99005 4.47599 4.96193 
bin_counts: 10 7 13 9 18 7 8 8 10 10 
scott@cmp:Homework-4$ mpiexec -n 6 ./mpi_histogram 10 0.0 5.0 100
bin_maxes: 0.588489 1.07443 1.56036 2.0463 2.53224 3.01818 3.50412 3.99005 4.47599 4.96193 
bin_counts: 10 7 13 9 18 7 8 8 10 10
scott@cmp:Homework-4$ mpiexec -n 5 ./mpi_histogram 10 0.0 5.0 100
bin_maxes: 0.588489 1.07443 1.56036 2.0463 2.53224 3.01818 3.50412 3.99005 4.47599 4.96193 
bin_counts: 10 7 13 9 18 7 8 8 10 10
scott@cmp:Homework-4$ 
scott@cmp:Homework-4$ mpiexec -n 4 ./mpi_histogram 10 0.0 5.0 100
bin_maxes: 0.588489 1.07443 1.56036 2.0463 2.53224 3.01818 3.50412 3.99005 4.47599 4.96193 
bin_counts: 10 7 13 9 18 7 8 8 10 10
scott@cmp:Homework-4$ 
*/

#include <iostream>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <algorithm>

//#define DEBUG
#ifdef DEBUG
    #define log(FMT, ...) printf(FMT, ##__VA_ARGS__)
    int DEBUG_c; //could be a char. Used to pause the program
    #define bp scanf("%c", &DEBUG_c)
#else
    #define bp
    #define log(...)
#endif // DEBUG



using namespace std;

int my_rank, comm_sz;
MPI_Comm comm;

//function copied&altered from mpi_many_msgs.c 
//designed to gather the expected lines from the command line.
void Get_arg(int argc, char* argv[], int* bin_count, float* min_meas, float* max_meas, int* data_count);

int main(int argc, char *argv[]) {
    // Initialize MPI, copied from mpi_many_msgs.c
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    //histogram variables
    int bin_count, data_count;
    float min_meas, max_meas;
    
    Get_arg(argc, argv, &bin_count, &min_meas, &max_meas, &data_count);

    //shared containers for histogram calculations. 
    vector<float> bin_maxes(bin_count);
    vector<int> bin_counts(bin_count, 0);
    vector<int> global_bin_counts(bin_count, 0);
    vector<float> data(data_count, 0);
    // Process 0 computes the bin maxes
    if (my_rank == 0) {
        //fill data with random floats between min_meas and max_meas

        srand(100);
        for (int i = 0; i < data_count; i++) {
            float temp = min_meas + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_meas - min_meas)));
            data[i] = temp;
        }
        //after data is generated, calculate the bin_maxes

        float bin_width = (*max_element(data.begin(), data.end()) - *min_element(data.begin(), data.end())) / bin_count;
        float min_val = *min_element(data.begin(), data.end());

        bin_maxes[0] = min_val + bin_width;
        for (int i = 1; i < bin_count-1; i++) {
            bin_maxes[i] = bin_maxes[i-1] + bin_width;
        }
        //can't be calculated, so a quick grab of the max element keeps the algorithm working.
        bin_maxes[bin_count-1] = *max_element(data.begin(), data.end());
    }

    // Broadcast bin_maxes to all processes
    MPI_Bcast(bin_maxes.data(), bin_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Determine how many data elements each process will get
    int base_size = data_count / comm_sz;
    int remainder = data_count % comm_sz;
    //adjust the size for situations where the data count isn't
    //evenly divisible by the number of processes.
    //this will make the first few processes have one more element.
    int my_size = base_size + (my_rank < remainder ? 1 : 0); 
    vector<float> local_data(my_size); 

    // Calculate displacements and send counts for MPI_Scatterv
    vector<int> send_counts(comm_sz);
    vector<int> displacements(comm_sz);
    int offset = 0;

    if (my_rank == 0) {
        for (int i = 0; i < comm_sz; i++) {
            send_counts[i] = base_size + (i < remainder ? 1 : 0);
            displacements[i] = offset;
            offset += send_counts[i];
        }
    }

    // Use scatter to get the points on the rubric... and for performance.
    //split the data by the counts and displacements. Then scatter to the 
    //local data vectors for each process.
    MPI_Scatterv(data.data(), send_counts.data(), displacements.data(), 
                MPI_FLOAT, local_data.data(), my_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Each process computes its local histogram counts
    for (float val : local_data) {
        for (int i = 0; i < bin_count; i++) {
            if (val <= bin_maxes[i]) {
                bin_counts[i]++; //binary search not worth implementing again.
                break;
            }
        }
    }
    //bin search copied from my homework 2, but was harder to do.
    // for(int i = threadID; i < num_data; i+=num_threads){
    //     int low = 0;
    //     int high = num_bins - 1;
    //     int mid;
    //     while(low <= high){
    //         mid = low+(high-low)/2;
    //         if(mid == 0 && data[i] <= bin_maxes[0]) {
    //             separatedBinCounts[threadID][0]++;
    //             break;
    //         }
    //         else if(data[i] <= bin_maxes[mid] && (mid == 0 || data[i] > bin_maxes[mid-1])) {
    //             separatedBinCounts[threadID][mid]++;
    //             break;
    //         } else if(data[i] > bin_maxes[mid]) {
    //             low = mid + 1;
    //         } else {
    //             high = mid - 1;
    //         }
    //     }

    /*=========================================================*/
    //AI assisted in writing this line of code:
    // Sum all local histograms into a global histogram at process 0
    MPI_Reduce(bin_counts.data(), global_bin_counts.data(), bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    /*=========================================================*/

    //print the processes
    if (my_rank == 0) {
        cout << "bin_maxes: ";
        for (float max : bin_maxes) {
            cout << max << " ";
        }
        cout << endl;

        cout << "bin_counts: ";
        for (int count : global_bin_counts) {
            cout << count << " ";
        }
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}

void Get_arg(int argc, char* argv[], int* bin_count, float* min_meas, float* max_meas, int* data_count) {
    if (my_rank == 0) {
        if (argc == 5) {
            *bin_count  = std::stoi(argv[1]);
            *min_meas   = std::stof(argv[2]);
            *max_meas   = std::stof(argv[3]);
            *data_count = std::stoi(argv[4]);
            if (*data_count <= 0) {
                std::cerr << "Data count must be greater than 0" << std::endl;
            }
        } else {
            std::cerr << "Usage: " << argv[0] << " <bin count> <min meas> <max meas> <data count>" << std::endl;
            *data_count = -1;
        }
    }
    // Broadcast all parameters to all processes
    MPI_Bcast(bin_count, 1, MPI_INT, 0, comm);
    MPI_Bcast(min_meas, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(max_meas, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(data_count, 1, MPI_INT, 0, comm);

    if (*data_count <= 0) {
        //kill program if usage isn't correct.
        MPI_Finalize();
        exit(0);
    }
}
