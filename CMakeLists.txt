
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


add_executable(Datagen src/Data_generator.cpp)
add_executable(Simulator_2 src/GL_Simulator_2.cpp)
add_library(SimLib src/Simulator.cpp src/Window.cpp src/PointMassSimulator.cpp src/SphereSimulator.cpp)

target_link_libraries(Datagen ${CONAN_LIBS} SimLib)
target_link_libraries(Simulator_2 ${CONAN_LIBS} SimLib)
target_link_libraries(SimLib ${CONAN_LIBS})

if(${UNIX})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,--stack-size=1000000")
elseif(${MSVC})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
else()
    message(FATAL_ERROR "Compiler and system are not supported")
endif()
