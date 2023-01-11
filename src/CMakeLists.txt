
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
        simlib/Window.cpp
        )
add_library(SimLib ${SimLibFiles})
target_include_directories(SimLib PUBLIC "${CMAKE_SOURCE_DIR}/include")


# Python Bindings
pybind11_add_module(CosmoSimPy MODULE cosmosimpy/CosmoSim.cpp)
target_link_libraries(CosmoSimPy PRIVATE SimLib)

add_executable(makeimage makeimage/makeimage.cpp)
add_executable(Simulator simulator/Simulator.cpp)

# Link Libraries
if (${host} MATCHES "idun.*")
    target_link_libraries(SimLib PUBLIC ${OpenCV_LIBS} ${LLIBS})
    target_link_libraries(makeimage PRIVATE SimLib)
    target_link_libraries(Simulator PRIVATE SimLib)
else ()
    target_link_libraries(SimLib PUBLIC ${CONAN_LIBS})
    target_link_libraries(makeimage PRIVATE SimLib)
    target_link_libraries(Simulator PRIVATE SimLib)
endif ()

# copy CosmoSimPy.*.pyd into Python directory
add_custom_command(TARGET CosmoSimPy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CosmoSimPy> "${CMAKE_SOURCE_DIR}/Python/CosmoSim"
        )
