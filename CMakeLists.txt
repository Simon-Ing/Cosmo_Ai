# (C) 2022: Hans Georg Schaathun <georg@schaathun.net>


message("CMakeLists.txt - CosmoSim - Starting ...")

cmake_minimum_required(VERSION 3.15)
project(CosmoAi LANGUAGES CXX VERSION 1.1.0)

option(COSMOSIM_BUILD_SIMULATOR "Build native cosmosim simulator executable" OFF)
option(COSMOSIM_BUILD_MAKEIMAGE "Build native cosmosim makeimage executable" OFF)

set(CMAKE_CXX_STANDARD 17)

# Place target artifacts into bin and lib folders
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


cmake_host_system_information(RESULT host QUERY HOSTNAME)
message("Running on ${host}")

# Symengine and OpenCV
if( ${host} MATCHES "idun.*")
	message("IDUN - conan is not available")
	set(LLIBS "symengine;gmp;mpfr;pthread")

	find_package( OpenCV REQUIRED )
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	find_package( Python COMPONENTS Development Interpreter ) 
	include_directories( ${Python_INCLUDE_DIRS} )

	message("LLIBS=${LLIBS}")
	message("OpenCV_LIBS=${OpenCV_LIBS}")
else()
      message("Including conan setup")
      include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
      conan_basic_setup()
endif()

include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.10.3
)
FetchContent_MakeAvailable(pybind11)
add_subdirectory("${CMAKE_BINARY_DIR}/_deps/pybind11-src")


add_subdirectory(src)


if(${UNIX})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,stack-size=1000000")
elseif(${MSVC})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
else()
    message(FATAL_ERROR "Compiler and system are not supported")
endif()

