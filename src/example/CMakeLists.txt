file(GLOB APP_SOURCES "*.cpp")
foreach (sourcefile ${APP_SOURCES})
    string(REGEX REPLACE ".+/" "" executablename1 ${sourcefile})
    string(REPLACE ".cpp" "" executablename ${executablename1})

    message("source files: " ${sourcefile} " executable: " ${executablename})
    add_executable(${executablename} ${sourcefile})
    target_link_libraries(${executablename} fp piqp pthread rt)
endforeach (sourcefile ${APP_SOURCES})
