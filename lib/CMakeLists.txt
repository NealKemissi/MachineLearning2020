cmake_minimum_required(VERSION 3.16)
project(lib)

set(CMAKE_CXX_STANDARD 20)

add_library(lib SHARED main.cpp)
add_library(lib_static STATIC main.cpp)