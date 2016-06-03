cmake_minimum_required(VERSION 3.4.1)
project(rethink_simd)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(Boost_USE_STATIC_LIBS       ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost 1.59.0 COMPONENTS program_options system)

set(SOURCE_FILES main.cpp)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
add_executable(rethink_simd ${SOURCE_FILES})
target_link_libraries(rethink_simd ${Boost_LIBRARIES})