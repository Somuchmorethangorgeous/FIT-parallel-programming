cmake_minimum_required(VERSION 3.17)
project(lab2 C)


set (CMAKE_C_FLAGS "-fopenmp ${CMAKE_C_FLAGS}")
set(CMAKE_C_STANDARD 99)

add_executable(lab2_v1 var1.c)
add_executable(lab2_v2 var2.c)

target_link_libraries(lab2_v1 m)
target_link_libraries(lab2_v2 m)