pybind11_add_module(pyfp "PyFp.cpp")
target_link_libraries(pyfp PRIVATE fpopt rt dl)