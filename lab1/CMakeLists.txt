cmake_minimum_required(VERSION 3.17)
project(lab1 C)

set(CMAKE_C_STANDARD 99)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_COMPILER mpicc)

add_executable(lab1_v1 var1.c)
add_executable(lab1_v2 var2.c)
target_link_libraries(lab1_v1 m)
target_link_libraries(lab1_v2 m)