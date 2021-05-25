#ifndef LAB5_EXECUTION_H
#define LAB5_EXECUTION_H

#include <mpi.h>
#include "cmath"
#include "../info/info.h"
#include "../threadArgs/threadArgs.h"


void doWork(Task task,  WorkingInfo* workingInfo);
int receiveTasks(int fromProc, ProcInfo* procInfo, WorkingInfo* workingInfo);
void initTaskList(ProcInfo* procInfo, WorkingInfo* workingInfo);
void* count(void* data);


#endif //LAB5_EXECUTION_H
