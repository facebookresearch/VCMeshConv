cmake_minimum_required(VERSION 3.10)
project(GraphSampling)

set(CMAKE_CXX_STANDARD 14)
find_package(ZLIB)



add_executable(GraphSampling main.cpp third-party/mdSArray.h third-party/mdVector.h third-party/mdVector.inl meshLoader.h meshPooler.h meshPooler_visualizer.h meshCNN.h third-party/cnpy/cnpy.cpp third-party/cnpy/cnpy.h)

target_link_libraries(GraphSampling ${ZLIB_LIBRARIES})

