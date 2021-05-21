#include "balance.h"

void* unloadOfTasks(void* data){
    int availabilityFlag;
    MPI_Status status;
    auto* tArgs = static_cast<thread_args*>(data);
    WorkingInfo* workingInfo = tArgs->workingInfo;

    while (workingInfo->globalIter < TOTAL_GLOBAL_ITER){
        MPI_Recv(&availabilityFlag, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        if(availabilityFlag == NO_TASKS_FLAG){
            return nullptr;
        }
        pthread_mutex_lock(&workingInfo->taskMutex);
        if(workingInfo->curTask > workingInfo->taskList.size() - TASKS_LEFT_TO_DO){
            availabilityFlag = 0;
        }
        pthread_mutex_unlock(&workingInfo->taskMutex);

        MPI_Send(&availabilityFlag, 1, MPI_INT, status.MPI_SOURCE, RESPONSE_TAG, MPI_COMM_WORLD);
        if(!availabilityFlag){
            continue;
        }

        pthread_mutex_lock(&workingInfo->taskMutex);
        Task taskToSend = workingInfo->taskList.back();
        workingInfo->taskList.pop_back();
        pthread_mutex_unlock(&workingInfo->taskMutex);

        MPI_Send(&taskToSend, sizeof(Task), MPI_BYTE, status.MPI_SOURCE, SEND_TASK_TAG, MPI_COMM_WORLD);
        ++workingInfo->tasksSend;
    }
    return nullptr;
}
