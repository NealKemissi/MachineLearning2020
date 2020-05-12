cmake_minimum_required(VERSION 3.16)
project(lib)

set(CMAKE_CXX_STANDARD 20)

set(EIGEN_DIR "C:\\Users\\christopher\\ProgramDev\\eigen-3.3.7")
include_directories(${EIGEN_DIR})

add_library(lib SHARED main.cpp)
add_library(lib_static STATIC main.cpp)
target_link_libraries(lib_static ${EIGEN_DIR})