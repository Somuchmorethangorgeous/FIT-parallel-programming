cmake_minimum_required(VERSION 3.17)
project(lab2 C)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set (CMAKE_C_FLAGS "-fopenmp -DDEBUG_INFO ${CMAKE_C_FLAGS}")
set(CMAKE_C_STANDARD 99)

add_executable(lab2 var1.c)
target_link_libraries(lab2 m)