file(GLOB test_srcs "*.cpp")

enable_testing()

add_executable(test.all ${test_srcs})
target_link_libraries(test.all fp piqp pthread rt)
