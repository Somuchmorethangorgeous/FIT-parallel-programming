#include "execution.h"


void doWork(Task task,  WorkingInfo* workingInfo){
    for (int i = 0; i < task.repeatNum; ++i){
        workingInfo->globalRes += sqrt(i);
    }
}


void initTaskList(ProcInfo* procInfo, WorkingInfo* workingInfo){
    workingInfo->taskList.clear();
    for (int i = 0; i < ITER_PER_PROC; ++i){
        workingInfo->taskList.emplace_back(abs(procInfo->rank - (workingInfo->globalIter % procInfo->size)));
    }
}


int receiveTasks(int fromProc, ProcInfo* procInfo, WorkingInfo* workingInfo){
    if (fromProc == procInfo->rank){
        return 0;
    }
    int availabilityFlag = 1;

    MPI_Send(&availabilityFlag,1,MPI_INT,fromProc, REQUEST_TAG, MPI_COMM_WORLD);
    MPI_Recv(&availabilityFlag,1,MPI_INT,fromProc, RESPONSE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    if (!availabilityFlag){
        return 0;
    }

    Task receivedTask;
    MPI_Recv(&receivedTask, sizeof(receivedTask), MPI_BYTE, fromProc, SEND_TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    pthread_mutex_lock(&workingInfo->taskMutex);
    workingInfo->taskList.push_back(receivedTask);
    pthread_mutex_unlock(&workingInfo->taskMutex);
    ++workingInfo->tasksReceived;
    return availabilityFlag;
}


void* count(void* data){
    int shiftProc;
    auto* tArgs = static_cast<thread_args*>(data);
    ProcInfo* procInfo = tArgs->procInfo;
    WorkingInfo* workingInfo = tArgs->workingInfo;

    for(workingInfo->globalIter = 0; workingInfo->globalIter < TOTAL_GLOBAL_ITER; ++workingInfo->globalIter, shiftProc = workingInfo->curTask = 0){
        workingInfo->tasksReceived = workingInfo->tasksSend = 0;
        initTaskList(procInfo, workingInfo);

        while (shiftProc < procInfo->size){
            for (; workingInfo->curTask < workingInfo->taskList.size(); ++workingInfo->curTask){
                pthread_mutex_unlock(&workingInfo->taskMutex);
                doWork(workingInfo->taskList[workingInfo->curTask], workingInfo);
                pthread_mutex_lock(&workingInfo->taskMutex);
            }

            pthread_mutex_unlock(&workingInfo->taskMutex);
            int requestProc = (procInfo->rank + shiftProc) % procInfo->size;
            if (receiveTasks(requestProc, procInfo, workingInfo) == 0){
                ++shiftProc;
            }
            pthread_mutex_lock(&workingInfo->taskMutex);
        }
        pthread_mutex_unlock(&workingInfo->taskMutex);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    int endFlag = NO_TASKS_FLAG;
    MPI_Send(&endFlag, 1, MPI_INT, procInfo->rank, REQUEST_TAG, MPI_COMM_WORLD);

    return nullptr;
}

