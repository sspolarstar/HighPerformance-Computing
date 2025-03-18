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

#include <iostream>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstdlib>

#define DEBUG

#ifdef DEBUG
    #define log(FMT, ...) printf(FMT, ##__VA_ARGS__)
    int DEBUG_c; //could be a char. Used to pause the program
    #define bp scanf("%c", &DEBUG_c)
#else
    #define bp
    #define log(...)
#endif // DEBUG

int      my_rank;
int      comm_sz;
MPI_Comm comm;

void Get_arg(int argc, char* argv[], int* bin_count, float* min_meas, float* max_meas, int* data_count);    

using namespace std;

int main(int argc, char *argv[])
{

    
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    // inputs: <bin count> <min meas> <max meas> <data count>
    int bin_count, data_count;
    float min_meas, max_meas;

    std::vector<int> bin_counts;
    std::vector<float> data, bin_maxes;

    Get_arg(argc, argv, &bin_count, &min_meas, &max_meas, &data_count);
    data.resize(data_count);
    bin_counts.resize(bin_count);

    if (my_rank == 0){ //generate the data
        srand(100);
        for (int i = 0; i < data_count; i++){
            data[i]  = min_meas + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_meas - min_meas)));
        }
        //calculate the bin maxes
        //in process 0, not specified to happen, but it lends itself
        //to serial execution.
        float bin_width = (*max_element(data.begin(), data.end()) - *min_element(data.begin(), data.end())) / bin_count;
        float min_val = *min_element(data.begin(), data.end());
        bin_maxes.resize(bin_count);        
        bin_maxes[0] = min_val + bin_width;
        for (int i = 1; i < bin_count-1; i++) {
            bin_maxes[i] = bin_maxes[i-1] + bin_width;
        }
        //can't be calculated, so a quick grab of the max element keeps the algorithm working.
        bin_maxes[bin_count-1] = *max_element(data.begin(), data.end());
    }
    //broadcast the data to all processes
    MPI_Bcast(&bin_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&min_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bin_maxes.data(), bin_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    

    if (my_rank != 0)
    {
        // data.resize(data_count); //reset sizes for all other processes, now.
    }
    
    //don't share full data array.
    // MPI_Bcast(data.data(), data_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //get ready for the reduction

    std::vector<int> global_bin_counts(bin_count);
    // MPI_Reduce(bin_counts.data(), global_bin_counts.data(), bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        for (int i = 0; i < bin_count; i++)
        {
            std::cout << bin_maxes[i] << " ";
        }
        std::cout << std::endl;

        for (int i = 0; i < bin_count; i++)
        {
            std::cout << global_bin_counts[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    exit(0);
}

//Copied from mpi_many_msgs.c
/*-------------------------------------------------------------------*/
void Get_arg(
    int    argc       /* in  */, 
    char*  argv[]     /* out */, 
    int*   bin_count  /* out */,
    float* min_meas   /* out */,
    float* max_meas   /* out */,
    int*   data_count /* out */
    ) {

 if (my_rank == 0) {
    if (argc == 5) {
        *bin_count = std::stoi(argv[1]);
        *min_meas   = std::stof(argv[2]);
        *max_meas   = std::stof(argv[3]);
        *data_count = std::stoi(argv[4]);
        if (*data_count <= 0) {
            std::cerr << "Data count must be greater than 0" << std::endl;
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " <bin count> <min meas> <max meas> <data count>" << std::endl;
       *data_count = -1; //should never be a negative number!
    }
 }

 MPI_Bcast(data_count, 1, MPI_INT, 0, comm); //wait for the broadcast from 0,
 if (*data_count <= 0) {
    MPI_Finalize();
    exit(0);
 }
}  /* Get_arg */