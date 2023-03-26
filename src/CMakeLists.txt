
set(SimLibFiles
        simlib/LensMap.cpp
        simlib/PureSampledLens.cpp
        simlib/PureSampledSISLens.cpp
        simlib/SampledRouletteLens.cpp
        simlib/SampledSISLens.cpp
        simlib/RouletteSISLens.cpp
        simlib/PMCLens.cpp
        simlib/RouletteLens.cpp
        simlib/EllipsoidSource.cpp
        simlib/LensModel.cpp
        simlib/PointMassLens.cpp
        simlib/RoulettePMLens.cpp
        simlib/simaux.cpp
        simlib/Source.cpp
        simlib/SphereLens.cpp
        simlib/SphericalSource.cpp
        simlib/TriangleSource.cpp
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


# copy CosmoSimPy.*.pyd into Python directory
add_custom_command(TARGET CosmoSimPy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CosmoSimPy> "${CMAKE_SOURCE_DIR}/Python/CosmoSim"
        )
