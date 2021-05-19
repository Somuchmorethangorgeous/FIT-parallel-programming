#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>

#define TOTAL_NUMBER_ITER_PER_PROC 4


int size, rank;
int curTask;
int globalIter;
double globalRes;

pthread_mutex_t taskMutex;

struct Task{
    int repeatNum = 0;

    explicit Task(int numberOftaks) : repeatNum(numberOftaks){}
};

std::vector<Task> taskList;


void errorInProgramm(){
    MPI_Finalize();
    abort();
}


void initTaskList(){
    taskList.clear();
    for (int i = 0; i < TOTAL_NUMBER_ITER_PER_PROC; ++i){
        taskList.emplace_back(abs(rank - (globalIter % size)));
    }
}


void doWork(Task task){
    for (int i = 0; i < task.repeatNum; ++i){
        globalRes += sqrt(i);
    }
}


void* count(void* data){
    for(globalIter = 0; globalIter < TOTAL_NUMBER_ITER_PER_PROC; ++globalIter, curTask = 0){
        pthread_mutex_lock(&taskMutex);
        initTaskList();

        for(;curTask < taskList.size();++curTask){
            pthread_mutex_unlock(&taskMutex);
            doWork(taskList[curTask]);
            pthread_mutex_lock(&taskMutex);
        }

        pthread_mutex_unlock(&taskMutex);

        MPI_Barrier(MPI_COMM_WORLD);
    }
    return nullptr;
}


void initThreads(){
    pthread_attr_t attrs;
    pthread_t executorThread;

    if(pthread_attr_init(&attrs)){
        std::cerr << "Cannot initialize attributes" << std::endl;
        errorInProgramm();
    }

    if(pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE)){
        std::cerr <<"Error in setting attributes" << std::endl;
        errorInProgramm();
    }

    pthread_mutex_init(&taskMutex, nullptr);

    if(pthread_create(&executorThread, nullptr, count, nullptr)){
        std::cerr << "Error in creating thread" << std::endl;
        errorInProgramm();
    }

    if(pthread_join(executorThread, nullptr)){
        std::cerr << "Cannot join a thread!" << std::endl;
        errorInProgramm();
    }


    pthread_attr_destroy(&attrs);
    pthread_mutex_destroy(&taskMutex);
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

    initThreads();

    MPI_Finalize();

    return 0;
}
