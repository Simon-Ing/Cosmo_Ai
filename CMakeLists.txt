
cmake_minimum_required(VERSION 3.15)
project(CosmoAi CXX)

# Download automatically, you can also just copy the conan.cmake file
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
            "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)


conan_cmake_run(REQUIRES
        symengine/0.9.0
        opencv/4.5.3
        BASIC_SETUP)


add_executable(Simulator_2 GL_Simulator_2.cpp)
add_library(SimLib Simulator.cpp)
add_executable(main main.cpp)
add_executable(Data data_generator.cpp)
add_executable(new new_data_gen.cpp)
add_executable(load load.cpp)

target_link_libraries(Simulator_2 ${CONAN_LIBS} SimLib)
target_link_libraries(SimLib ${CONAN_LIBS} ${GINAC_LIBRARIES})
target_link_libraries(main ${CONAN_LIBS} )
target_link_libraries(Data ${CONAN_LIBS} )
target_link_libraries(new ${CONAN_LIBS} )

target_include_directories(SimLib PRIVATE ${GINAC_INCLUDE_DIRS})
target_include_directories(load PRIVATE ${GINAC_INCLUDE_DIRS})
target_link_libraries(load PRIVATE ${GINAC_LIBRARIES} ${CONAN_LIBS})