#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>
#define TOTAL_NUMBER_ITER_PER_PROC 4


int size, rank;
int curIter = 0;
int globalIter = 0;
double globalRes = 0;

struct Task{
    int repeatNum = 0;

    explicit Task(int numberOftaks) : repeatNum(numberOftaks){}
};

std::vector<Task> taskList;


void initTaskList(){
    taskList.clear();
    for (int i = 0; i < TOTAL_NUMBER_ITER_PER_PROC; ++i){
        taskList.emplace_back(abs(rank - (curIter % size)));
    }
}

void doWork(Task task){
    for (int i = 0; i < task.repeatNum; ++i){
        globalRes += sqrt(i);
    }
}

int main(int argc, char **argv) {
    int provided_level;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_level);
    if (provided_level != MPI_THREAD_MULTIPLE) {
        std::cerr << "ERROR: In this version there is not implemented a required level of multithreading!" << std::endl;
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    MPI_Finalize();

    return 0;
}
