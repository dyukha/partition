project(partition)

set(CMAKE_CXX_FLAGS "-O2 -std=c++17 -Wall -Wextra -fopenmp")

set(SOURCE_FILES
     src/Runner.cpp
#     src/GradientDescent.cpp
#     src/GradientDescentImpl.cpp
#     src/GradientDescentManyClusters.cpp
#     src/Graph.cpp
#     src/Partition.cpp
#     src/Projections.cpp
#     src/RecursiveClustering.cpp
#     src/utils.cpp
)

add_executable(partition ${SOURCE_FILES})