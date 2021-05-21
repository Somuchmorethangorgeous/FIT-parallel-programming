#include <iostream>
#include <mpi.h>
#include "balance/balance.h"
#include "execution/execution.h"


void errorInProgramm(){
    MPI_Finalize();
    abort();
}


void initThreads(ProcInfo& procInfo){
    pthread_attr_t attrs;
    pthread_t executorThread;
    pthread_t receiverThread;
    WorkingInfo workingInfo;
    auto* tArgs = new thread_args(&procInfo, &workingInfo);

    if(pthread_attr_init(&attrs)){
        std::cerr << "Cannot initialize attributes" << std::endl;
        errorInProgramm();
    }

    if(pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE)){
        std::cerr << "Error in setting attributes" << std::endl;
        errorInProgramm();
    }

    pthread_mutex_init(&workingInfo.taskMutex, nullptr);

    if(pthread_create(&receiverThread, nullptr, unloadOfTasks, (void*)tArgs)){
        std::cerr << "Error in creating thread" << std::endl;
        errorInProgramm();
    }

    if(pthread_create(&executorThread, nullptr, count, (void*)tArgs)){
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
    pthread_mutex_destroy(&workingInfo.taskMutex);
}


int main(int argc, char **argv) {
    int provided_level;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_level);
    if (provided_level != MPI_THREAD_MULTIPLE) {
        std::cerr << "ERROR: In this version there is not implemented a required level of multithreading!" << std::endl;
        return -1;
    }

    ProcInfo procInfo;
    MPI_Comm_rank(MPI_COMM_WORLD, &procInfo.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procInfo.size);

    initThreads(procInfo);

    MPI_Finalize();

    return 0;
}
