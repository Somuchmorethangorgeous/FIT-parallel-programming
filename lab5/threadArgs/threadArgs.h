#ifndef LAB5_THREADARGS_H
#define LAB5_THREADARGS_H

struct thread_args{
    ProcInfo* procInfo;
    WorkingInfo* workingInfo;
    thread_args(ProcInfo* pInfo, WorkingInfo* wInfo) : procInfo(pInfo), workingInfo(wInfo){}
};


#endif //LAB5_THREADARGS_H
