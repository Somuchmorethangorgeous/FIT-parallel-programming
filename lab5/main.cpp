#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>

#define TOTAL_NUMBER_ITER_PER_PROC 4
#define TASKS_LEFT_TO_DO 5
#define REQUEST_TAG 333
#define RESPONSE_TAG 33
#define SEND_TASK_TAG 3


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
    double start, end;
    for(globalIter = 0; globalIter < TOTAL_NUMBER_ITER_PER_PROC; ++globalIter, curTask = 0){
        pthread_mutex_lock(&taskMutex);
        initTaskList();

        start = MPI_Wtime();
        for(;curTask < taskList.size();++curTask){
            pthread_mutex_unlock(&taskMutex);
            doWork(taskList[curTask]);
            pthread_mutex_lock(&taskMutex);
        }

        pthread_mutex_unlock(&taskMutex);
        end = MPI_Wtime();
        std::cout << "proc: " << rank << " time: "  << end - start << " at iter: " << globalIter << " tasks implemented: " << curTask << std::endl;

        if (rank == 0){
            std::cout << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    return nullptr;
}

void* unloadOfTasks(void* data){
    int availabilityFlag;
    MPI_Status status;
    while (globalIter < TOTAL_NUMBER_ITER_PER_PROC){

        pthread_mutex_lock(&taskMutex);
        if(curTask > taskList.size() - TASKS_LEFT_TO_DO){
            availabilityFlag = 0;
        }
        pthread_mutex_unlock(&taskMutex);

        MPI_Send(&availabilityFlag, 1, MPI_INT, status.MPI_SOURCE, RESPONSE_TAG, MPI_COMM_WORLD);
        if(!availabilityFlag){
            continue;
        }

        Task taskToSend = taskList.back();
        taskList.pop_back();
        MPI_Send(&taskToSend, sizeof(Task), MPI_BYTE, status.MPI_SOURCE, SEND_TASK_TAG, MPI_COMM_WORLD);
    }
    return nullptr;
}

void initThreads(){
    pthread_attr_t attrs;
    pthread_t executorThread;
    pthread_t receiverThread;

    if(pthread_attr_init(&attrs)){
        std::cerr << "Cannot initialize attributes" << std::endl;
        errorInProgramm();
    }

    if(pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE)){
        std::cerr <<"Error in setting attributes" << std::endl;
        errorInProgramm();
    }

    pthread_mutex_init(&taskMutex, nullptr);

    if(pthread_create(&receiverThread, nullptr, unloadOfTasks, nullptr)){
        std::cerr << "Error in creating thread" << std::endl;
        errorInProgramm();
    }

    if(pthread_create(&executorThread, nullptr, count, nullptr)){
        std::cerr << "Error in creating thread" << std::endl;
        errorInProgramm();
    }

    if(pthread_join(executorThread, nullptr)){
        std::cerr << "Cannot join a thread!" << std::endl;
        errorInProgramm();
    }

    if(pthread_join(receiverThread, nullptr)){
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
