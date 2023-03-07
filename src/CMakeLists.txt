
set(SimLibFiles
        simlib/EllipsoidSource.cpp
        simlib/LensModel.cpp
        simlib/PointMassLens.cpp
        simlib/RoulettePMLens.cpp
        simlib/simaux.cpp
        simlib/Source.cpp
        simlib/SphereLens.cpp
        simlib/SphericalSource.cpp
        simlib/TriangleSource.cpp
        simlib/PixMapLens.cpp
        simlib/PMCLens.cpp
        simlib/RoulettePMCLens.cpp
        )

add_library(SimLib ${SimLibFiles})
set_target_properties(SimLib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(SimLib PUBLIC "${CMAKE_SOURCE_DIR}/include")
if (${host} MATCHES "idun.*")
    target_link_libraries(SimLib PUBLIC ${OpenCV_LIBS} ${LLIBS})
else ()
    target_link_libraries(SimLib PUBLIC ${CONAN_LIBS})
endif ()


# Python Bindings
pybind11_add_module(CosmoSimPy MODULE cosmosimpy/CosmoSim.cpp)
target_link_libraries(CosmoSimPy PRIVATE SimLib)


if (COSMOSIM_BUILD_MAKEIMAGE)
    add_executable(makeimage makeimage/makeimage.cpp)
    target_link_libraries(makeimage PRIVATE SimLib)
endif()

if (COSMOSIM_BUILD_SIMULATOR)
    target_link_libraries(Simulator PRIVATE SimLib)
    add_executable(Simulator simulator/Simulator.cpp simulator/Window.cpp)
endif()


# copy CosmoSimPy.*.pyd into Python directory
add_custom_command(TARGET CosmoSimPy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CosmoSimPy> "${CMAKE_SOURCE_DIR}/Python/CosmoSim"
        )
