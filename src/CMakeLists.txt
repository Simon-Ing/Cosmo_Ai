
set(SimLibFiles
        simlib/SampledLens.cpp
        simlib/SampledPsiFunctionLens.cpp
        simlib/Lens.cpp
	simlib/ClusterLens.cpp
        simlib/PsiFunctionLens.cpp
        simlib/SIS.cpp
        simlib/SIE.cpp
        simlib/RaytraceModel.cpp
        simlib/RouletteModel.cpp
        simlib/EllipsoidSource.cpp
        simlib/ImageSource.cpp
        simlib/SimulatorModel.cpp
        simlib/PointMass.cpp
        simlib/RouletteRegenerator.cpp
        simlib/simaux.cpp
        simlib/Source.cpp
        simlib/SphericalSource.cpp
        simlib/TriangleSource.cpp
        simlib/SourceConstellation.cpp
        )
     # Currently unused:
     #    simlib/LensMap.cpp
     #    simlib/PMCLens.cpp

add_library(SimLib ${SimLibFiles})
set_target_properties(SimLib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(SimLib PUBLIC BEFORE "${CMAKE_SOURCE_DIR}/include" ${pybind11_INCLUDE_DIRS})
target_link_libraries(SimLib ${OpenCV_LIBS} symengine)

# Python Bindings
pybind11_add_module(CosmoSimPy MODULE CosmoSim/bindings.cpp )
target_link_libraries(CosmoSimPy PRIVATE SimLib)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
message("IPO ${supported}")

if( supported )
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET SimLib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET CosmoSimPy PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()

install( TARGETS CosmoSimPy DESTINATION CosmoSim/ )
message( "end of src/CMakeLists.txt" )

# copy CosmoSimPy.*.pyd into Python directory
# add_custom_command(TARGET CosmoSimPy POST_BUILD
# COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CosmoSimPy> "${CMAKE_SOURCE_DIR}/CosmoSimPy/CosmoSim"
# )
