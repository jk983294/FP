cmake_minimum_required(VERSION 3.5)

project(fp)

set(CMAKE_INSTALL_PREFIX "/opt/version/latest/fp/")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/)

find_package(OpenMP REQUIRED)

SET(CMAKE_CXX_FLAGS "-std=c++17 -O3 -s -Wall -Werror")
#SET(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall -Werror")
#SET(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall -Werror -fsanitize=address -fno-omit-frame-pointer")

message ("cxx Flags: " ${CMAKE_CXX_FLAGS})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)

include_directories(SYSTEM /opt/3rd/common/include)
include_directories(SYSTEM /opt/3rd/Eigen)
include_directories(include)
include_directories(/opt/version/latest/zergtool/include)

link_directories(/opt/3rd/common/lib)

# install(DIRECTORY include/ DESTINATION include)
install(FILES include/fp_opt.h DESTINATION include)

add_subdirectory(src)
add_subdirectory(pybind11)
add_subdirectory(pyfp)
