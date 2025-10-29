cmake_minimum_required(VERSION 3.5)

project(fp)

set(CMAKE_INSTALL_PREFIX "/opt/version/latest/fp/")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message ("OpenMP_CXX_FOUND: " ${OpenMP_CXX_FOUND})
    message ("${OpenMP_CXX_LIBRARIES}: " ${OpenMP_CXX_LIBRARIES})
    message ("${OpenMP_CXX_FLAGS}: " ${OpenMP_CXX_FLAGS})
endif()

# SET(CMAKE_CXX_FLAGS "-std=c++17 -O3 -s -Wall -Werror")
SET(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall -Werror")
#SET(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall -Werror -fsanitize=address -fno-omit-frame-pointer")

message ("cxx Flags: " ${CMAKE_CXX_FLAGS})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)

add_definitions(-DPIQP_STD_OPTIONAL)

include_directories(SYSTEM /opt/3rd/common/include)
include_directories(SYSTEM /opt/3rd/Eigen)
include_directories(include)
include_directories(/opt/version/latest/zerg/include)
#include_directories(/opt/version/latest/ornate/include)

link_directories(/opt/3rd/common/lib)

# install(DIRECTORY include/ DESTINATION include)
install(FILES include/fp_opt.h DESTINATION include)

add_subdirectory(src)
 add_subdirectory(pybind11)
 add_subdirectory(pyfp)
