cmake_minimum_required(VERSION 3.16)
project(project)

set(CMAKE_CXX_STANDARD 20)
link_directories("C:\\Users\\Neal\\CLionProjects\\MachineLearning2020\\lib\\cmake-build-debug")

add_executable(project main.cpp)
target_link_libraries(project lib_static.lib)