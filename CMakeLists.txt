# (C) 2022: Hans Georg Schaathun <georg@schaathun.net>


message("CMakeLists.txt - CosmoSim - Starting ...")

cmake_minimum_required(VERSION 3.15)
project(CosmoAi LANGUAGES CXX VERSION 1.1.0)

add_definitions("-std=c++11")

cmake_host_system_information(RESULT host QUERY HOSTNAME)
# set(host "foobar")
message("Running on ${host}")

# Symengine and OpenCV
if( ${host} MATCHES "idun.*")
	message("IDUN - conan is not available")
	set(LLIBS "symengine;gmp;mpfr;pthread")

	find_package( OpenCV REQUIRED )
	include_directories( ${OpenCV_INCLUDE_DIRS} )

	message("LLIBS=${LLIBS}")
	message("OpenCV_LIBS=${OpenCV_LIBS}")
else()
      message("Including conan setup")
      include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
      conan_basic_setup()
endif()


add_executable(makeimage src/makeimage.cpp)
# add_executable(testmask src/testmask.cpp)
add_executable(Simulator src/Simulator.cpp)
add_library(SimLib 
   src/simaux.cpp 
   src/Source.cpp src/SphericalSource.cpp src/EllipsoidSource.cpp
   src/TriangleSource.cpp
   src/LensModel.cpp src/PointMassLens.cpp src/SphereLens.cpp 
   src/RoulettePMLens.cpp
   src/Window.cpp)


if( ${host} MATCHES "idun.*")
	target_link_libraries(makeimage ${OpenCV_LIBS} ${LLIBS} SimLib)
	# target_link_libraries(testmask ${OpenCV_LIBS} ${LLIBS} SimLib)
	target_link_libraries(Simulator ${OpenCV_LIBS} ${LLIBS} SimLib)
	target_link_libraries(SimLib ${OpenCV_LIBS} ${LLIBS})
else()
   target_link_libraries(makeimage ${CONAN_LIBS} SimLib)
   # target_link_libraries(testmask ${CONAN_LIBS} SimLib)
   target_link_libraries(Simulator ${CONAN_LIBS} SimLib)
   target_link_libraries(SimLib ${CONAN_LIBS})
endif()

# Python Bindings
find_package(pybind11 REQUIRED)
pybind11_add_module(CosmoSim MODULE src/CosmoSim.cpp ${SimLibFiles})
target_link_libraries(CosmoSim PRIVATE ${CONAN_LIBS})

if(${UNIX})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,stack-size=1000000")
elseif(${MSVC})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
else()
    message(FATAL_ERROR "Compiler and system are not supported")
endif()

install( TARGETS makeimage Simulator DESTINATION bin)
install( TARGETS SimLib CosmoSim DESTINATION lib)
