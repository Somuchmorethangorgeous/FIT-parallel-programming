#ifndef LAB5_INFO_H
#define LAB5_INFO_H

#include <vector>
#define TOTAL_GLOBAL_ITER 100
#define ITER_PER_PROC 2e7
#define TASKS_LEFT_TO_DO 5
#define NO_TASKS_FLAG -1
#define REQUEST_TAG 333
#define RESPONSE_TAG 33
#define SEND_TASK_TAG 3


struct Task{
    int repeatNum = 0;

    explicit Task(int numberOftaks) : repeatNum(numberOftaks){}
    Task() : Task(0){};
};


struct ProcInfo{
    int size;
    int rank;
};


struct WorkingInfo{
    int curTask = 0;
    int globalIter = 0;
    int tasksSend = 0;
    int tasksReceived = 0;
    double globalRes = 0.0;
    std::vector<Task> taskList;
    pthread_mutex_t taskMutex;
};


#endif //LAB5_INFO_H
