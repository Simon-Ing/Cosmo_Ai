
message("CMakeLists.txt - CosmoSim - Starting ...")

cmake_minimum_required(VERSION 3.15)
project(CosmoAi CXX)

# cmake_host_system_information(RESULT host QUERY HOSTNAME)
set(host "foobar")

message("Running on ${host}")
if( ${host} MATCHES "idun.*")
   message("Running on IDUN.  We will not use Conan")
else()
   message("CONAN setup")

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

endif()

add_executable(makeimage src/makeimage.cpp)
add_executable(Simulator src/Simulator.cpp)
add_library(SimLib 
   src/simaux.cpp 
   src/Source.cpp src/SphericalSource.cpp src/EllipsoidSource.cpp
   src/TriangleSource.cpp
   src/LensModel.cpp src/PointMassLens.cpp src/SphereLens.cpp 
   src/RoulettePMLens.cpp
   src/Window.cpp)

if( ${host} MATCHES "idun.*")
   set(HOME "/cluster/home/hasc/")
   message( "Homedir is ${HOME}" )
   set(CMAKE_LIBRARY_PATH "${HOME}/local/usr/lib64")

   find_library(SYMLIB NAMES symengine)
   # find_library(CVLIB NAMES opencv_core)
   # find_library(CVILIB NAMES opencv_imgcodecs)
   set(CVLIB "opencv")
   
   message( "CVLIB=${CVLIB}" )
   # find_library(NAMES opencv PATHS "${HOME}/local/usr/lib64")
   target_link_libraries(makeimage ${SYMLIB} ${CVLIB} SimLib)
   target_link_libraries(Simulator ${SYMLIB} ${CVLIB} SimLib)
   target_link_libraries(SimLib ${SYMLIB} ${CVLIB})
else()
   target_link_libraries(makeimage ${CONAN_LIBS} SimLib)
   target_link_libraries(Simulator ${CONAN_LIBS} SimLib)
   target_link_libraries(SimLib ${CONAN_LIBS})
endif()

if(${UNIX})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,stack-size=1000000")
elseif(${MSVC})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
else()
    message(FATAL_ERROR "Compiler and system are not supported")
endif()
