include(CMakeParseArguments)

macro(conan_find_apple_frameworks FRAMEWORKS_FOUND FRAMEWORKS SUFFIX BUILD_TYPE)
    if(APPLE)
        if(CMAKE_BUILD_TYPE)
            set(_BTYPE ${CMAKE_BUILD_TYPE})
        elseif(NOT BUILD_TYPE STREQUAL "")
            set(_BTYPE ${BUILD_TYPE})
        endif()
        if(_BTYPE)
            if(${_BTYPE} MATCHES "Debug|_DEBUG")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_DEBUG} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_DEBUG} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            elseif(${_BTYPE} MATCHES "Release|_RELEASE")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_RELEASE} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_RELEASE} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            elseif(${_BTYPE} MATCHES "RelWithDebInfo|_RELWITHDEBINFO")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_RELWITHDEBINFO} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_RELWITHDEBINFO} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            elseif(${_BTYPE} MATCHES "MinSizeRel|_MINSIZEREL")
                set(CONAN_FRAMEWORKS${SUFFIX} ${CONAN_FRAMEWORKS${SUFFIX}_MINSIZEREL} ${CONAN_FRAMEWORKS${SUFFIX}})
                set(CONAN_FRAMEWORK_DIRS${SUFFIX} ${CONAN_FRAMEWORK_DIRS${SUFFIX}_MINSIZEREL} ${CONAN_FRAMEWORK_DIRS${SUFFIX}})
            endif()
        endif()
        foreach(_FRAMEWORK ${FRAMEWORKS})
            # https://cmake.org/pipermail/cmake-developers/2017-August/030199.html
            find_library(CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND NAME ${_FRAMEWORK} PATHS ${CONAN_FRAMEWORK_DIRS${SUFFIX}} CMAKE_FIND_ROOT_PATH_BOTH)
            if(CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND)
                list(APPEND ${FRAMEWORKS_FOUND} ${CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND})
            else()
                message(FATAL_ERROR "Framework library ${_FRAMEWORK} not found in paths: ${CONAN_FRAMEWORK_DIRS${SUFFIX}}")
            endif()
        endforeach()
    endif()
endmacro()


#################
###  OPENCV
#################
set(CONAN_OPENCV_ROOT "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b")
set(CONAN_INCLUDE_DIRS_OPENCV "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/include"
			"/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/include/opencv4")
set(CONAN_LIB_DIRS_OPENCV "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/lib")
set(CONAN_BIN_DIRS_OPENCV "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/bin")
set(CONAN_RES_DIRS_OPENCV "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/res")
set(CONAN_SRC_DIRS_OPENCV )
set(CONAN_BUILD_DIRS_OPENCV "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/"
			"/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_OPENCV )
set(CONAN_LIBS_OPENCV opencv_ml opencv_photo opencv_highgui opencv_videoio opencv_imgcodecs opencv_objdetect opencv_stitching opencv_dnn opencv_gapi opencv_video opencv_calib3d opencv_features2d opencv_flann opencv_imgproc opencv_core)
set(CONAN_PKG_LIBS_OPENCV opencv_ml opencv_photo opencv_highgui opencv_videoio opencv_imgcodecs opencv_objdetect opencv_stitching opencv_dnn opencv_gapi opencv_video opencv_calib3d opencv_features2d opencv_flann opencv_imgproc opencv_core)
set(CONAN_SYSTEM_LIBS_OPENCV dl m pthread rt)
set(CONAN_FRAMEWORKS_OPENCV )
set(CONAN_FRAMEWORKS_FOUND_OPENCV "")  # Will be filled later
set(CONAN_DEFINES_OPENCV )
set(CONAN_BUILD_MODULES_PATHS_OPENCV )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_OPENCV )

set(CONAN_C_FLAGS_OPENCV "")
set(CONAN_CXX_FLAGS_OPENCV "")
set(CONAN_SHARED_LINKER_FLAGS_OPENCV "")
set(CONAN_EXE_LINKER_FLAGS_OPENCV "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_OPENCV_LIST "")
set(CONAN_CXX_FLAGS_OPENCV_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_OPENCV_LIST "")
set(CONAN_EXE_LINKER_FLAGS_OPENCV_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_OPENCV "${CONAN_FRAMEWORKS_OPENCV}" "_OPENCV" "")
# Append to aggregated values variable
set(CONAN_LIBS_OPENCV ${CONAN_PKG_LIBS_OPENCV} ${CONAN_SYSTEM_LIBS_OPENCV} ${CONAN_FRAMEWORKS_FOUND_OPENCV})


#################
###  JASPER
#################
set(CONAN_JASPER_ROOT "/home/simon/.conan/data/jasper/2.0.32/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b")
set(CONAN_INCLUDE_DIRS_JASPER "/home/simon/.conan/data/jasper/2.0.32/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/include")
set(CONAN_LIB_DIRS_JASPER "/home/simon/.conan/data/jasper/2.0.32/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/lib")
set(CONAN_BIN_DIRS_JASPER )
set(CONAN_RES_DIRS_JASPER )
set(CONAN_SRC_DIRS_JASPER )
set(CONAN_BUILD_DIRS_JASPER "/home/simon/.conan/data/jasper/2.0.32/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/"
			"/home/simon/.conan/data/jasper/2.0.32/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_JASPER )
set(CONAN_LIBS_JASPER jasper)
set(CONAN_PKG_LIBS_JASPER jasper)
set(CONAN_SYSTEM_LIBS_JASPER m)
set(CONAN_FRAMEWORKS_JASPER )
set(CONAN_FRAMEWORKS_FOUND_JASPER "")  # Will be filled later
set(CONAN_DEFINES_JASPER )
set(CONAN_BUILD_MODULES_PATHS_JASPER )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_JASPER )

set(CONAN_C_FLAGS_JASPER "")
set(CONAN_CXX_FLAGS_JASPER "")
set(CONAN_SHARED_LINKER_FLAGS_JASPER "")
set(CONAN_EXE_LINKER_FLAGS_JASPER "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_JASPER_LIST "")
set(CONAN_CXX_FLAGS_JASPER_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_JASPER_LIST "")
set(CONAN_EXE_LINKER_FLAGS_JASPER_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_JASPER "${CONAN_FRAMEWORKS_JASPER}" "_JASPER" "")
# Append to aggregated values variable
set(CONAN_LIBS_JASPER ${CONAN_PKG_LIBS_JASPER} ${CONAN_SYSTEM_LIBS_JASPER} ${CONAN_FRAMEWORKS_FOUND_JASPER})


#################
###  OPENEXR
#################
set(CONAN_OPENEXR_ROOT "/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84")
set(CONAN_INCLUDE_DIRS_OPENEXR "/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/include"
			"/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/include/OpenEXR")
set(CONAN_LIB_DIRS_OPENEXR "/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/lib")
set(CONAN_BIN_DIRS_OPENEXR )
set(CONAN_RES_DIRS_OPENEXR )
set(CONAN_SRC_DIRS_OPENEXR )
set(CONAN_BUILD_DIRS_OPENEXR "/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/"
			"/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_OPENEXR )
set(CONAN_LIBS_OPENEXR IlmImfUtil-2_5 IlmImf-2_5 Imath-2_5 Half-2_5 IexMath-2_5 IlmThread-2_5 Iex-2_5)
set(CONAN_PKG_LIBS_OPENEXR IlmImfUtil-2_5 IlmImf-2_5 Imath-2_5 Half-2_5 IexMath-2_5 IlmThread-2_5 Iex-2_5)
set(CONAN_SYSTEM_LIBS_OPENEXR stdc++ pthread)
set(CONAN_FRAMEWORKS_OPENEXR )
set(CONAN_FRAMEWORKS_FOUND_OPENEXR "")  # Will be filled later
set(CONAN_DEFINES_OPENEXR )
set(CONAN_BUILD_MODULES_PATHS_OPENEXR )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_OPENEXR )

set(CONAN_C_FLAGS_OPENEXR "")
set(CONAN_CXX_FLAGS_OPENEXR "")
set(CONAN_SHARED_LINKER_FLAGS_OPENEXR "")
set(CONAN_EXE_LINKER_FLAGS_OPENEXR "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_OPENEXR_LIST "")
set(CONAN_CXX_FLAGS_OPENEXR_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_OPENEXR_LIST "")
set(CONAN_EXE_LINKER_FLAGS_OPENEXR_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_OPENEXR "${CONAN_FRAMEWORKS_OPENEXR}" "_OPENEXR" "")
# Append to aggregated values variable
set(CONAN_LIBS_OPENEXR ${CONAN_PKG_LIBS_OPENEXR} ${CONAN_SYSTEM_LIBS_OPENEXR} ${CONAN_FRAMEWORKS_FOUND_OPENEXR})


#################
###  LIBTIFF
#################
set(CONAN_LIBTIFF_ROOT "/home/simon/.conan/data/libtiff/4.3.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18")
set(CONAN_INCLUDE_DIRS_LIBTIFF "/home/simon/.conan/data/libtiff/4.3.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18/include")
set(CONAN_LIB_DIRS_LIBTIFF "/home/simon/.conan/data/libtiff/4.3.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18/lib")
set(CONAN_BIN_DIRS_LIBTIFF )
set(CONAN_RES_DIRS_LIBTIFF )
set(CONAN_SRC_DIRS_LIBTIFF )
set(CONAN_BUILD_DIRS_LIBTIFF "/home/simon/.conan/data/libtiff/4.3.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18/")
set(CONAN_FRAMEWORK_DIRS_LIBTIFF )
set(CONAN_LIBS_LIBTIFF tiffxx tiff)
set(CONAN_PKG_LIBS_LIBTIFF tiffxx tiff)
set(CONAN_SYSTEM_LIBS_LIBTIFF m)
set(CONAN_FRAMEWORKS_LIBTIFF )
set(CONAN_FRAMEWORKS_FOUND_LIBTIFF "")  # Will be filled later
set(CONAN_DEFINES_LIBTIFF )
set(CONAN_BUILD_MODULES_PATHS_LIBTIFF )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBTIFF )

set(CONAN_C_FLAGS_LIBTIFF "")
set(CONAN_CXX_FLAGS_LIBTIFF "")
set(CONAN_SHARED_LINKER_FLAGS_LIBTIFF "")
set(CONAN_EXE_LINKER_FLAGS_LIBTIFF "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBTIFF_LIST "")
set(CONAN_CXX_FLAGS_LIBTIFF_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBTIFF_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBTIFF_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBTIFF "${CONAN_FRAMEWORKS_LIBTIFF}" "_LIBTIFF" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBTIFF ${CONAN_PKG_LIBS_LIBTIFF} ${CONAN_SYSTEM_LIBS_LIBTIFF} ${CONAN_FRAMEWORKS_FOUND_LIBTIFF})


#################
###  EIGEN
#################
set(CONAN_EIGEN_ROOT "/home/simon/.conan/data/eigen/3.3.9/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_EIGEN "/home/simon/.conan/data/eigen/3.3.9/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3")
set(CONAN_LIB_DIRS_EIGEN )
set(CONAN_BIN_DIRS_EIGEN )
set(CONAN_RES_DIRS_EIGEN )
set(CONAN_SRC_DIRS_EIGEN )
set(CONAN_BUILD_DIRS_EIGEN "/home/simon/.conan/data/eigen/3.3.9/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/")
set(CONAN_FRAMEWORK_DIRS_EIGEN )
set(CONAN_LIBS_EIGEN )
set(CONAN_PKG_LIBS_EIGEN )
set(CONAN_SYSTEM_LIBS_EIGEN m)
set(CONAN_FRAMEWORKS_EIGEN )
set(CONAN_FRAMEWORKS_FOUND_EIGEN "")  # Will be filled later
set(CONAN_DEFINES_EIGEN )
set(CONAN_BUILD_MODULES_PATHS_EIGEN )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_EIGEN )

set(CONAN_C_FLAGS_EIGEN "")
set(CONAN_CXX_FLAGS_EIGEN "")
set(CONAN_SHARED_LINKER_FLAGS_EIGEN "")
set(CONAN_EXE_LINKER_FLAGS_EIGEN "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_EIGEN_LIST "")
set(CONAN_CXX_FLAGS_EIGEN_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_EIGEN_LIST "")
set(CONAN_EXE_LINKER_FLAGS_EIGEN_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_EIGEN "${CONAN_FRAMEWORKS_EIGEN}" "_EIGEN" "")
# Append to aggregated values variable
set(CONAN_LIBS_EIGEN ${CONAN_PKG_LIBS_EIGEN} ${CONAN_SYSTEM_LIBS_EIGEN} ${CONAN_FRAMEWORKS_FOUND_EIGEN})


#################
###  FFMPEG
#################
set(CONAN_FFMPEG_ROOT "/home/simon/.conan/data/ffmpeg/4.4/_/_/package/2770b46b8645e6555ada38043cbb4bf19de6ac0f")
set(CONAN_INCLUDE_DIRS_FFMPEG "/home/simon/.conan/data/ffmpeg/4.4/_/_/package/2770b46b8645e6555ada38043cbb4bf19de6ac0f/include")
set(CONAN_LIB_DIRS_FFMPEG "/home/simon/.conan/data/ffmpeg/4.4/_/_/package/2770b46b8645e6555ada38043cbb4bf19de6ac0f/lib")
set(CONAN_BIN_DIRS_FFMPEG "/home/simon/.conan/data/ffmpeg/4.4/_/_/package/2770b46b8645e6555ada38043cbb4bf19de6ac0f/bin")
set(CONAN_RES_DIRS_FFMPEG )
set(CONAN_SRC_DIRS_FFMPEG )
set(CONAN_BUILD_DIRS_FFMPEG "/home/simon/.conan/data/ffmpeg/4.4/_/_/package/2770b46b8645e6555ada38043cbb4bf19de6ac0f/")
set(CONAN_FRAMEWORK_DIRS_FFMPEG )
set(CONAN_LIBS_FFMPEG avdevice avfilter avformat avcodec swscale swresample postproc avutil)
set(CONAN_PKG_LIBS_FFMPEG avdevice avfilter avformat avcodec swscale swresample postproc avutil)
set(CONAN_SYSTEM_LIBS_FFMPEG m pthread dl)
set(CONAN_FRAMEWORKS_FFMPEG )
set(CONAN_FRAMEWORKS_FOUND_FFMPEG "")  # Will be filled later
set(CONAN_DEFINES_FFMPEG )
set(CONAN_BUILD_MODULES_PATHS_FFMPEG )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_FFMPEG )

set(CONAN_C_FLAGS_FFMPEG "")
set(CONAN_CXX_FLAGS_FFMPEG "")
set(CONAN_SHARED_LINKER_FLAGS_FFMPEG "-Wl,-Bsymbolic")
set(CONAN_EXE_LINKER_FLAGS_FFMPEG "-Wl,-Bsymbolic")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_FFMPEG_LIST "")
set(CONAN_CXX_FLAGS_FFMPEG_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_FFMPEG_LIST "-Wl,-Bsymbolic")
set(CONAN_EXE_LINKER_FLAGS_FFMPEG_LIST "-Wl,-Bsymbolic")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_FFMPEG "${CONAN_FRAMEWORKS_FFMPEG}" "_FFMPEG" "")
# Append to aggregated values variable
set(CONAN_LIBS_FFMPEG ${CONAN_PKG_LIBS_FFMPEG} ${CONAN_SYSTEM_LIBS_FFMPEG} ${CONAN_FRAMEWORKS_FOUND_FFMPEG})


#################
###  QUIRC
#################
set(CONAN_QUIRC_ROOT "/home/simon/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711")
set(CONAN_INCLUDE_DIRS_QUIRC "/home/simon/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711/include")
set(CONAN_LIB_DIRS_QUIRC "/home/simon/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711/lib")
set(CONAN_BIN_DIRS_QUIRC )
set(CONAN_RES_DIRS_QUIRC )
set(CONAN_SRC_DIRS_QUIRC )
set(CONAN_BUILD_DIRS_QUIRC "/home/simon/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711/")
set(CONAN_FRAMEWORK_DIRS_QUIRC )
set(CONAN_LIBS_QUIRC quirc)
set(CONAN_PKG_LIBS_QUIRC quirc)
set(CONAN_SYSTEM_LIBS_QUIRC m)
set(CONAN_FRAMEWORKS_QUIRC )
set(CONAN_FRAMEWORKS_FOUND_QUIRC "")  # Will be filled later
set(CONAN_DEFINES_QUIRC )
set(CONAN_BUILD_MODULES_PATHS_QUIRC )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_QUIRC )

set(CONAN_C_FLAGS_QUIRC "")
set(CONAN_CXX_FLAGS_QUIRC "")
set(CONAN_SHARED_LINKER_FLAGS_QUIRC "")
set(CONAN_EXE_LINKER_FLAGS_QUIRC "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_QUIRC_LIST "")
set(CONAN_CXX_FLAGS_QUIRC_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_QUIRC_LIST "")
set(CONAN_EXE_LINKER_FLAGS_QUIRC_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_QUIRC "${CONAN_FRAMEWORKS_QUIRC}" "_QUIRC" "")
# Append to aggregated values variable
set(CONAN_LIBS_QUIRC ${CONAN_PKG_LIBS_QUIRC} ${CONAN_SYSTEM_LIBS_QUIRC} ${CONAN_FRAMEWORKS_FOUND_QUIRC})


#################
###  GTK
#################
set(CONAN_GTK_ROOT "/home/simon/.conan/data/gtk/system/_/_/package/cc5e5ca8b619d2dffaaeae1e4924ae3ce085a8f1")
set(CONAN_INCLUDE_DIRS_GTK "/usr/include/gtk-2.0"
			"/usr/lib/x86_64-linux-gnu/gtk-2.0/include"
			"/usr/include/pango-1.0"
			"/usr/include/atk-1.0"
			"/usr/include/gdk-pixbuf-2.0"
			"/usr/include/libmount"
			"/usr/include/blkid"
			"/usr/include/pango-1.0"
			"/usr/include/fribidi"
			"/usr/include/cairo"
			"/usr/include/pixman-1"
			"/usr/include/harfbuzz"
			"/usr/include/glib-2.0"
			"/usr/lib/x86_64-linux-gnu/glib-2.0/include"
			"/usr/include/uuid"
			"/usr/include/freetype2"
			"/usr/include/libpng16")
set(CONAN_LIB_DIRS_GTK )
set(CONAN_BIN_DIRS_GTK )
set(CONAN_RES_DIRS_GTK )
set(CONAN_SRC_DIRS_GTK )
set(CONAN_BUILD_DIRS_GTK "/home/simon/.conan/data/gtk/system/_/_/package/cc5e5ca8b619d2dffaaeae1e4924ae3ce085a8f1/")
set(CONAN_FRAMEWORK_DIRS_GTK )
set(CONAN_LIBS_GTK )
set(CONAN_PKG_LIBS_GTK )
set(CONAN_SYSTEM_LIBS_GTK gtk-x11-2.0 gdk-x11-2.0 pangocairo-1.0 atk-1.0 cairo gdk_pixbuf-2.0 gio-2.0 pangoft2-1.0 pango-1.0 gobject-2.0 glib-2.0 harfbuzz fontconfig freetype)
set(CONAN_FRAMEWORKS_GTK )
set(CONAN_FRAMEWORKS_FOUND_GTK "")  # Will be filled later
set(CONAN_DEFINES_GTK )
set(CONAN_BUILD_MODULES_PATHS_GTK )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_GTK )

set(CONAN_C_FLAGS_GTK "-pthread")
set(CONAN_CXX_FLAGS_GTK "-pthread")
set(CONAN_SHARED_LINKER_FLAGS_GTK "")
set(CONAN_EXE_LINKER_FLAGS_GTK "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_GTK_LIST "-pthread")
set(CONAN_CXX_FLAGS_GTK_LIST "-pthread")
set(CONAN_SHARED_LINKER_FLAGS_GTK_LIST "")
set(CONAN_EXE_LINKER_FLAGS_GTK_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_GTK "${CONAN_FRAMEWORKS_GTK}" "_GTK" "")
# Append to aggregated values variable
set(CONAN_LIBS_GTK ${CONAN_PKG_LIBS_GTK} ${CONAN_SYSTEM_LIBS_GTK} ${CONAN_FRAMEWORKS_FOUND_GTK})


#################
###  PROTOBUF
#################
set(CONAN_PROTOBUF_ROOT "/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2")
set(CONAN_INCLUDE_DIRS_PROTOBUF "/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/include")
set(CONAN_LIB_DIRS_PROTOBUF "/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/lib")
set(CONAN_BIN_DIRS_PROTOBUF "/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/bin")
set(CONAN_RES_DIRS_PROTOBUF )
set(CONAN_SRC_DIRS_PROTOBUF )
set(CONAN_BUILD_DIRS_PROTOBUF "/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/"
			"/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/lib/cmake/protobuf")
set(CONAN_FRAMEWORK_DIRS_PROTOBUF )
set(CONAN_LIBS_PROTOBUF protoc protobuf)
set(CONAN_PKG_LIBS_PROTOBUF protoc protobuf)
set(CONAN_SYSTEM_LIBS_PROTOBUF pthread)
set(CONAN_FRAMEWORKS_PROTOBUF )
set(CONAN_FRAMEWORKS_FOUND_PROTOBUF "")  # Will be filled later
set(CONAN_DEFINES_PROTOBUF )
set(CONAN_BUILD_MODULES_PATHS_PROTOBUF "/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/lib/cmake/protobuf/protobuf-generate.cmake"
			"/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/lib/cmake/protobuf/protobuf-module.cmake"
			"/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/lib/cmake/protobuf/protobuf-options.cmake")
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_PROTOBUF )

set(CONAN_C_FLAGS_PROTOBUF "")
set(CONAN_CXX_FLAGS_PROTOBUF "")
set(CONAN_SHARED_LINKER_FLAGS_PROTOBUF "")
set(CONAN_EXE_LINKER_FLAGS_PROTOBUF "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_PROTOBUF_LIST "")
set(CONAN_CXX_FLAGS_PROTOBUF_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_PROTOBUF_LIST "")
set(CONAN_EXE_LINKER_FLAGS_PROTOBUF_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_PROTOBUF "${CONAN_FRAMEWORKS_PROTOBUF}" "_PROTOBUF" "")
# Append to aggregated values variable
set(CONAN_LIBS_PROTOBUF ${CONAN_PKG_LIBS_PROTOBUF} ${CONAN_SYSTEM_LIBS_PROTOBUF} ${CONAN_FRAMEWORKS_FOUND_PROTOBUF})


#################
###  ADE
#################
set(CONAN_ADE_ROOT "/home/simon/.conan/data/ade/0.1.1f/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56")
set(CONAN_INCLUDE_DIRS_ADE "/home/simon/.conan/data/ade/0.1.1f/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/include")
set(CONAN_LIB_DIRS_ADE "/home/simon/.conan/data/ade/0.1.1f/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/lib")
set(CONAN_BIN_DIRS_ADE )
set(CONAN_RES_DIRS_ADE )
set(CONAN_SRC_DIRS_ADE )
set(CONAN_BUILD_DIRS_ADE "/home/simon/.conan/data/ade/0.1.1f/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/"
			"/home/simon/.conan/data/ade/0.1.1f/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_ADE )
set(CONAN_LIBS_ADE ade)
set(CONAN_PKG_LIBS_ADE ade)
set(CONAN_SYSTEM_LIBS_ADE )
set(CONAN_FRAMEWORKS_ADE )
set(CONAN_FRAMEWORKS_FOUND_ADE "")  # Will be filled later
set(CONAN_DEFINES_ADE )
set(CONAN_BUILD_MODULES_PATHS_ADE )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_ADE )

set(CONAN_C_FLAGS_ADE "")
set(CONAN_CXX_FLAGS_ADE "")
set(CONAN_SHARED_LINKER_FLAGS_ADE "")
set(CONAN_EXE_LINKER_FLAGS_ADE "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_ADE_LIST "")
set(CONAN_CXX_FLAGS_ADE_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_ADE_LIST "")
set(CONAN_EXE_LINKER_FLAGS_ADE_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_ADE "${CONAN_FRAMEWORKS_ADE}" "_ADE" "")
# Append to aggregated values variable
set(CONAN_LIBS_ADE ${CONAN_PKG_LIBS_ADE} ${CONAN_SYSTEM_LIBS_ADE} ${CONAN_FRAMEWORKS_FOUND_ADE})


#################
###  LIBJPEG
#################
set(CONAN_LIBJPEG_ROOT "/home/simon/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_LIBJPEG "/home/simon/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(CONAN_LIB_DIRS_LIBJPEG "/home/simon/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_LIBJPEG )
set(CONAN_RES_DIRS_LIBJPEG "/home/simon/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/res")
set(CONAN_SRC_DIRS_LIBJPEG )
set(CONAN_BUILD_DIRS_LIBJPEG "/home/simon/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/")
set(CONAN_FRAMEWORK_DIRS_LIBJPEG )
set(CONAN_LIBS_LIBJPEG jpeg)
set(CONAN_PKG_LIBS_LIBJPEG jpeg)
set(CONAN_SYSTEM_LIBS_LIBJPEG )
set(CONAN_FRAMEWORKS_LIBJPEG )
set(CONAN_FRAMEWORKS_FOUND_LIBJPEG "")  # Will be filled later
set(CONAN_DEFINES_LIBJPEG "-DLIBJPEG_STATIC")
set(CONAN_BUILD_MODULES_PATHS_LIBJPEG )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBJPEG "LIBJPEG_STATIC")

set(CONAN_C_FLAGS_LIBJPEG "")
set(CONAN_CXX_FLAGS_LIBJPEG "")
set(CONAN_SHARED_LINKER_FLAGS_LIBJPEG "")
set(CONAN_EXE_LINKER_FLAGS_LIBJPEG "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBJPEG_LIST "")
set(CONAN_CXX_FLAGS_LIBJPEG_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBJPEG_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBJPEG_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBJPEG "${CONAN_FRAMEWORKS_LIBJPEG}" "_LIBJPEG" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBJPEG ${CONAN_PKG_LIBS_LIBJPEG} ${CONAN_SYSTEM_LIBS_LIBJPEG} ${CONAN_FRAMEWORKS_FOUND_LIBJPEG})


#################
###  LIBDEFLATE
#################
set(CONAN_LIBDEFLATE_ROOT "/home/simon/.conan/data/libdeflate/1.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_LIBDEFLATE "/home/simon/.conan/data/libdeflate/1.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(CONAN_LIB_DIRS_LIBDEFLATE "/home/simon/.conan/data/libdeflate/1.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_LIBDEFLATE )
set(CONAN_RES_DIRS_LIBDEFLATE )
set(CONAN_SRC_DIRS_LIBDEFLATE )
set(CONAN_BUILD_DIRS_LIBDEFLATE "/home/simon/.conan/data/libdeflate/1.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/")
set(CONAN_FRAMEWORK_DIRS_LIBDEFLATE )
set(CONAN_LIBS_LIBDEFLATE deflate)
set(CONAN_PKG_LIBS_LIBDEFLATE deflate)
set(CONAN_SYSTEM_LIBS_LIBDEFLATE )
set(CONAN_FRAMEWORKS_LIBDEFLATE )
set(CONAN_FRAMEWORKS_FOUND_LIBDEFLATE "")  # Will be filled later
set(CONAN_DEFINES_LIBDEFLATE )
set(CONAN_BUILD_MODULES_PATHS_LIBDEFLATE )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBDEFLATE )

set(CONAN_C_FLAGS_LIBDEFLATE "")
set(CONAN_CXX_FLAGS_LIBDEFLATE "")
set(CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE "")
set(CONAN_EXE_LINKER_FLAGS_LIBDEFLATE "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBDEFLATE_LIST "")
set(CONAN_CXX_FLAGS_LIBDEFLATE_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBDEFLATE_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBDEFLATE "${CONAN_FRAMEWORKS_LIBDEFLATE}" "_LIBDEFLATE" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBDEFLATE ${CONAN_PKG_LIBS_LIBDEFLATE} ${CONAN_SYSTEM_LIBS_LIBDEFLATE} ${CONAN_FRAMEWORKS_FOUND_LIBDEFLATE})


#################
###  XZ_UTILS
#################
set(CONAN_XZ_UTILS_ROOT "/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_XZ_UTILS "/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(CONAN_LIB_DIRS_XZ_UTILS "/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_XZ_UTILS "/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/bin")
set(CONAN_RES_DIRS_XZ_UTILS )
set(CONAN_SRC_DIRS_XZ_UTILS )
set(CONAN_BUILD_DIRS_XZ_UTILS "/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_XZ_UTILS )
set(CONAN_LIBS_XZ_UTILS lzma)
set(CONAN_PKG_LIBS_XZ_UTILS lzma)
set(CONAN_SYSTEM_LIBS_XZ_UTILS pthread)
set(CONAN_FRAMEWORKS_XZ_UTILS )
set(CONAN_FRAMEWORKS_FOUND_XZ_UTILS "")  # Will be filled later
set(CONAN_DEFINES_XZ_UTILS "-DLZMA_API_STATIC")
set(CONAN_BUILD_MODULES_PATHS_XZ_UTILS )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_XZ_UTILS "LZMA_API_STATIC")

set(CONAN_C_FLAGS_XZ_UTILS "")
set(CONAN_CXX_FLAGS_XZ_UTILS "")
set(CONAN_SHARED_LINKER_FLAGS_XZ_UTILS "")
set(CONAN_EXE_LINKER_FLAGS_XZ_UTILS "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_XZ_UTILS_LIST "")
set(CONAN_CXX_FLAGS_XZ_UTILS_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_LIST "")
set(CONAN_EXE_LINKER_FLAGS_XZ_UTILS_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_XZ_UTILS "${CONAN_FRAMEWORKS_XZ_UTILS}" "_XZ_UTILS" "")
# Append to aggregated values variable
set(CONAN_LIBS_XZ_UTILS ${CONAN_PKG_LIBS_XZ_UTILS} ${CONAN_SYSTEM_LIBS_XZ_UTILS} ${CONAN_FRAMEWORKS_FOUND_XZ_UTILS})


#################
###  JBIG
#################
set(CONAN_JBIG_ROOT "/home/simon/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304")
set(CONAN_INCLUDE_DIRS_JBIG "/home/simon/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/include")
set(CONAN_LIB_DIRS_JBIG "/home/simon/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/lib")
set(CONAN_BIN_DIRS_JBIG "/home/simon/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/bin")
set(CONAN_RES_DIRS_JBIG )
set(CONAN_SRC_DIRS_JBIG )
set(CONAN_BUILD_DIRS_JBIG "/home/simon/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/")
set(CONAN_FRAMEWORK_DIRS_JBIG )
set(CONAN_LIBS_JBIG jbig)
set(CONAN_PKG_LIBS_JBIG jbig)
set(CONAN_SYSTEM_LIBS_JBIG )
set(CONAN_FRAMEWORKS_JBIG )
set(CONAN_FRAMEWORKS_FOUND_JBIG "")  # Will be filled later
set(CONAN_DEFINES_JBIG )
set(CONAN_BUILD_MODULES_PATHS_JBIG )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_JBIG )

set(CONAN_C_FLAGS_JBIG "")
set(CONAN_CXX_FLAGS_JBIG "")
set(CONAN_SHARED_LINKER_FLAGS_JBIG "")
set(CONAN_EXE_LINKER_FLAGS_JBIG "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_JBIG_LIST "")
set(CONAN_CXX_FLAGS_JBIG_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_JBIG_LIST "")
set(CONAN_EXE_LINKER_FLAGS_JBIG_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_JBIG "${CONAN_FRAMEWORKS_JBIG}" "_JBIG" "")
# Append to aggregated values variable
set(CONAN_LIBS_JBIG ${CONAN_PKG_LIBS_JBIG} ${CONAN_SYSTEM_LIBS_JBIG} ${CONAN_FRAMEWORKS_FOUND_JBIG})


#################
###  ZSTD
#################
set(CONAN_ZSTD_ROOT "/home/simon/.conan/data/zstd/1.5.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_ZSTD "/home/simon/.conan/data/zstd/1.5.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(CONAN_LIB_DIRS_ZSTD "/home/simon/.conan/data/zstd/1.5.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_ZSTD )
set(CONAN_RES_DIRS_ZSTD )
set(CONAN_SRC_DIRS_ZSTD )
set(CONAN_BUILD_DIRS_ZSTD "/home/simon/.conan/data/zstd/1.5.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/")
set(CONAN_FRAMEWORK_DIRS_ZSTD )
set(CONAN_LIBS_ZSTD zstd)
set(CONAN_PKG_LIBS_ZSTD zstd)
set(CONAN_SYSTEM_LIBS_ZSTD pthread)
set(CONAN_FRAMEWORKS_ZSTD )
set(CONAN_FRAMEWORKS_FOUND_ZSTD "")  # Will be filled later
set(CONAN_DEFINES_ZSTD )
set(CONAN_BUILD_MODULES_PATHS_ZSTD )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_ZSTD )

set(CONAN_C_FLAGS_ZSTD "")
set(CONAN_CXX_FLAGS_ZSTD "")
set(CONAN_SHARED_LINKER_FLAGS_ZSTD "")
set(CONAN_EXE_LINKER_FLAGS_ZSTD "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_ZSTD_LIST "")
set(CONAN_CXX_FLAGS_ZSTD_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_ZSTD_LIST "")
set(CONAN_EXE_LINKER_FLAGS_ZSTD_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_ZSTD "${CONAN_FRAMEWORKS_ZSTD}" "_ZSTD" "")
# Append to aggregated values variable
set(CONAN_LIBS_ZSTD ${CONAN_PKG_LIBS_ZSTD} ${CONAN_SYSTEM_LIBS_ZSTD} ${CONAN_FRAMEWORKS_FOUND_ZSTD})


#################
###  LIBWEBP
#################
set(CONAN_LIBWEBP_ROOT "/home/simon/.conan/data/libwebp/1.2.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656")
set(CONAN_INCLUDE_DIRS_LIBWEBP "/home/simon/.conan/data/libwebp/1.2.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656/include")
set(CONAN_LIB_DIRS_LIBWEBP "/home/simon/.conan/data/libwebp/1.2.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656/lib")
set(CONAN_BIN_DIRS_LIBWEBP )
set(CONAN_RES_DIRS_LIBWEBP )
set(CONAN_SRC_DIRS_LIBWEBP )
set(CONAN_BUILD_DIRS_LIBWEBP "/home/simon/.conan/data/libwebp/1.2.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656/")
set(CONAN_FRAMEWORK_DIRS_LIBWEBP )
set(CONAN_LIBS_LIBWEBP webpdecoder webpdemux webpmux webp)
set(CONAN_PKG_LIBS_LIBWEBP webpdecoder webpdemux webpmux webp)
set(CONAN_SYSTEM_LIBS_LIBWEBP pthread m)
set(CONAN_FRAMEWORKS_LIBWEBP )
set(CONAN_FRAMEWORKS_FOUND_LIBWEBP "")  # Will be filled later
set(CONAN_DEFINES_LIBWEBP )
set(CONAN_BUILD_MODULES_PATHS_LIBWEBP )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBWEBP )

set(CONAN_C_FLAGS_LIBWEBP "")
set(CONAN_CXX_FLAGS_LIBWEBP "")
set(CONAN_SHARED_LINKER_FLAGS_LIBWEBP "")
set(CONAN_EXE_LINKER_FLAGS_LIBWEBP "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBWEBP_LIST "")
set(CONAN_CXX_FLAGS_LIBWEBP_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBWEBP_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBWEBP_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBWEBP "${CONAN_FRAMEWORKS_LIBWEBP}" "_LIBWEBP" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBWEBP ${CONAN_PKG_LIBS_LIBWEBP} ${CONAN_SYSTEM_LIBS_LIBWEBP} ${CONAN_FRAMEWORKS_FOUND_LIBWEBP})


#################
###  FREETYPE
#################
set(CONAN_FREETYPE_ROOT "/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3")
set(CONAN_INCLUDE_DIRS_FREETYPE "/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/include"
			"/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/include/freetype2")
set(CONAN_LIB_DIRS_FREETYPE "/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/lib")
set(CONAN_BIN_DIRS_FREETYPE "/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/bin")
set(CONAN_RES_DIRS_FREETYPE "/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/res")
set(CONAN_SRC_DIRS_FREETYPE )
set(CONAN_BUILD_DIRS_FREETYPE "/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/"
			"/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_FREETYPE )
set(CONAN_LIBS_FREETYPE freetype)
set(CONAN_PKG_LIBS_FREETYPE freetype)
set(CONAN_SYSTEM_LIBS_FREETYPE m)
set(CONAN_FRAMEWORKS_FREETYPE )
set(CONAN_FRAMEWORKS_FOUND_FREETYPE "")  # Will be filled later
set(CONAN_DEFINES_FREETYPE )
set(CONAN_BUILD_MODULES_PATHS_FREETYPE )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_FREETYPE )

set(CONAN_C_FLAGS_FREETYPE "")
set(CONAN_CXX_FLAGS_FREETYPE "")
set(CONAN_SHARED_LINKER_FLAGS_FREETYPE "")
set(CONAN_EXE_LINKER_FLAGS_FREETYPE "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_FREETYPE_LIST "")
set(CONAN_CXX_FLAGS_FREETYPE_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_FREETYPE_LIST "")
set(CONAN_EXE_LINKER_FLAGS_FREETYPE_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_FREETYPE "${CONAN_FRAMEWORKS_FREETYPE}" "_FREETYPE" "")
# Append to aggregated values variable
set(CONAN_LIBS_FREETYPE ${CONAN_PKG_LIBS_FREETYPE} ${CONAN_SYSTEM_LIBS_FREETYPE} ${CONAN_FRAMEWORKS_FOUND_FREETYPE})


#################
###  OPENJPEG
#################
set(CONAN_OPENJPEG_ROOT "/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_OPENJPEG "/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include/openjpeg-2.4")
set(CONAN_LIB_DIRS_OPENJPEG "/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_OPENJPEG )
set(CONAN_RES_DIRS_OPENJPEG )
set(CONAN_SRC_DIRS_OPENJPEG )
set(CONAN_BUILD_DIRS_OPENJPEG "/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_OPENJPEG )
set(CONAN_LIBS_OPENJPEG openjp2)
set(CONAN_PKG_LIBS_OPENJPEG openjp2)
set(CONAN_SYSTEM_LIBS_OPENJPEG pthread m)
set(CONAN_FRAMEWORKS_OPENJPEG )
set(CONAN_FRAMEWORKS_FOUND_OPENJPEG "")  # Will be filled later
set(CONAN_DEFINES_OPENJPEG )
set(CONAN_BUILD_MODULES_PATHS_OPENJPEG )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_OPENJPEG )

set(CONAN_C_FLAGS_OPENJPEG "")
set(CONAN_CXX_FLAGS_OPENJPEG "")
set(CONAN_SHARED_LINKER_FLAGS_OPENJPEG "")
set(CONAN_EXE_LINKER_FLAGS_OPENJPEG "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_OPENJPEG_LIST "")
set(CONAN_CXX_FLAGS_OPENJPEG_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_OPENJPEG_LIST "")
set(CONAN_EXE_LINKER_FLAGS_OPENJPEG_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_OPENJPEG "${CONAN_FRAMEWORKS_OPENJPEG}" "_OPENJPEG" "")
# Append to aggregated values variable
set(CONAN_LIBS_OPENJPEG ${CONAN_PKG_LIBS_OPENJPEG} ${CONAN_SYSTEM_LIBS_OPENJPEG} ${CONAN_FRAMEWORKS_FOUND_OPENJPEG})


#################
###  OPENH264
#################
set(CONAN_OPENH264_ROOT "/home/simon/.conan/data/openh264/2.1.1/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56")
set(CONAN_INCLUDE_DIRS_OPENH264 "/home/simon/.conan/data/openh264/2.1.1/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/include")
set(CONAN_LIB_DIRS_OPENH264 "/home/simon/.conan/data/openh264/2.1.1/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/lib")
set(CONAN_BIN_DIRS_OPENH264 )
set(CONAN_RES_DIRS_OPENH264 )
set(CONAN_SRC_DIRS_OPENH264 )
set(CONAN_BUILD_DIRS_OPENH264 "/home/simon/.conan/data/openh264/2.1.1/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/")
set(CONAN_FRAMEWORK_DIRS_OPENH264 )
set(CONAN_LIBS_OPENH264 openh264)
set(CONAN_PKG_LIBS_OPENH264 openh264)
set(CONAN_SYSTEM_LIBS_OPENH264 m pthread stdc++)
set(CONAN_FRAMEWORKS_OPENH264 )
set(CONAN_FRAMEWORKS_FOUND_OPENH264 "")  # Will be filled later
set(CONAN_DEFINES_OPENH264 )
set(CONAN_BUILD_MODULES_PATHS_OPENH264 )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_OPENH264 )

set(CONAN_C_FLAGS_OPENH264 "")
set(CONAN_CXX_FLAGS_OPENH264 "")
set(CONAN_SHARED_LINKER_FLAGS_OPENH264 "")
set(CONAN_EXE_LINKER_FLAGS_OPENH264 "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_OPENH264_LIST "")
set(CONAN_CXX_FLAGS_OPENH264_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_OPENH264_LIST "")
set(CONAN_EXE_LINKER_FLAGS_OPENH264_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_OPENH264 "${CONAN_FRAMEWORKS_OPENH264}" "_OPENH264" "")
# Append to aggregated values variable
set(CONAN_LIBS_OPENH264 ${CONAN_PKG_LIBS_OPENH264} ${CONAN_SYSTEM_LIBS_OPENH264} ${CONAN_FRAMEWORKS_FOUND_OPENH264})


#################
###  LIBX264
#################
set(CONAN_LIBX264_ROOT "/home/simon/.conan/data/libx264/20191217/_/_/package/52bfaf5780e9ee7fba3fa4a492156dbcf0dff057")
set(CONAN_INCLUDE_DIRS_LIBX264 "/home/simon/.conan/data/libx264/20191217/_/_/package/52bfaf5780e9ee7fba3fa4a492156dbcf0dff057/include")
set(CONAN_LIB_DIRS_LIBX264 "/home/simon/.conan/data/libx264/20191217/_/_/package/52bfaf5780e9ee7fba3fa4a492156dbcf0dff057/lib")
set(CONAN_BIN_DIRS_LIBX264 )
set(CONAN_RES_DIRS_LIBX264 )
set(CONAN_SRC_DIRS_LIBX264 )
set(CONAN_BUILD_DIRS_LIBX264 "/home/simon/.conan/data/libx264/20191217/_/_/package/52bfaf5780e9ee7fba3fa4a492156dbcf0dff057/")
set(CONAN_FRAMEWORK_DIRS_LIBX264 )
set(CONAN_LIBS_LIBX264 x264)
set(CONAN_PKG_LIBS_LIBX264 x264)
set(CONAN_SYSTEM_LIBS_LIBX264 dl pthread m)
set(CONAN_FRAMEWORKS_LIBX264 )
set(CONAN_FRAMEWORKS_FOUND_LIBX264 "")  # Will be filled later
set(CONAN_DEFINES_LIBX264 )
set(CONAN_BUILD_MODULES_PATHS_LIBX264 )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBX264 )

set(CONAN_C_FLAGS_LIBX264 "")
set(CONAN_CXX_FLAGS_LIBX264 "")
set(CONAN_SHARED_LINKER_FLAGS_LIBX264 "")
set(CONAN_EXE_LINKER_FLAGS_LIBX264 "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBX264_LIST "")
set(CONAN_CXX_FLAGS_LIBX264_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBX264_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBX264_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBX264 "${CONAN_FRAMEWORKS_LIBX264}" "_LIBX264" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBX264 ${CONAN_PKG_LIBS_LIBX264} ${CONAN_SYSTEM_LIBS_LIBX264} ${CONAN_FRAMEWORKS_FOUND_LIBX264})


#################
###  LIBX265
#################
set(CONAN_LIBX265_ROOT "/home/simon/.conan/data/libx265/3.4/_/_/package/75e903d40ade00f9adc18d07c1d36ecb3e595a39")
set(CONAN_INCLUDE_DIRS_LIBX265 "/home/simon/.conan/data/libx265/3.4/_/_/package/75e903d40ade00f9adc18d07c1d36ecb3e595a39/include")
set(CONAN_LIB_DIRS_LIBX265 "/home/simon/.conan/data/libx265/3.4/_/_/package/75e903d40ade00f9adc18d07c1d36ecb3e595a39/lib")
set(CONAN_BIN_DIRS_LIBX265 )
set(CONAN_RES_DIRS_LIBX265 )
set(CONAN_SRC_DIRS_LIBX265 )
set(CONAN_BUILD_DIRS_LIBX265 "/home/simon/.conan/data/libx265/3.4/_/_/package/75e903d40ade00f9adc18d07c1d36ecb3e595a39/")
set(CONAN_FRAMEWORK_DIRS_LIBX265 )
set(CONAN_LIBS_LIBX265 x265)
set(CONAN_PKG_LIBS_LIBX265 x265)
set(CONAN_SYSTEM_LIBS_LIBX265 dl pthread m stdc++)
set(CONAN_FRAMEWORKS_LIBX265 )
set(CONAN_FRAMEWORKS_FOUND_LIBX265 "")  # Will be filled later
set(CONAN_DEFINES_LIBX265 )
set(CONAN_BUILD_MODULES_PATHS_LIBX265 )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBX265 )

set(CONAN_C_FLAGS_LIBX265 "")
set(CONAN_CXX_FLAGS_LIBX265 "")
set(CONAN_SHARED_LINKER_FLAGS_LIBX265 "-Wl,-Bsymbolic,-znoexecstack")
set(CONAN_EXE_LINKER_FLAGS_LIBX265 "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBX265_LIST "")
set(CONAN_CXX_FLAGS_LIBX265_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBX265_LIST "-Wl,-Bsymbolic,-znoexecstack")
set(CONAN_EXE_LINKER_FLAGS_LIBX265_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBX265 "${CONAN_FRAMEWORKS_LIBX265}" "_LIBX265" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBX265 ${CONAN_PKG_LIBS_LIBX265} ${CONAN_SYSTEM_LIBS_LIBX265} ${CONAN_FRAMEWORKS_FOUND_LIBX265})


#################
###  LIBVPX
#################
set(CONAN_LIBVPX_ROOT "/home/simon/.conan/data/libvpx/1.10.0/_/_/package/21d176c6d550e07ab7b17c1513a8ddb7fb960145")
set(CONAN_INCLUDE_DIRS_LIBVPX "/home/simon/.conan/data/libvpx/1.10.0/_/_/package/21d176c6d550e07ab7b17c1513a8ddb7fb960145/include")
set(CONAN_LIB_DIRS_LIBVPX "/home/simon/.conan/data/libvpx/1.10.0/_/_/package/21d176c6d550e07ab7b17c1513a8ddb7fb960145/lib")
set(CONAN_BIN_DIRS_LIBVPX )
set(CONAN_RES_DIRS_LIBVPX )
set(CONAN_SRC_DIRS_LIBVPX )
set(CONAN_BUILD_DIRS_LIBVPX "/home/simon/.conan/data/libvpx/1.10.0/_/_/package/21d176c6d550e07ab7b17c1513a8ddb7fb960145/")
set(CONAN_FRAMEWORK_DIRS_LIBVPX )
set(CONAN_LIBS_LIBVPX vpx)
set(CONAN_PKG_LIBS_LIBVPX vpx)
set(CONAN_SYSTEM_LIBS_LIBVPX )
set(CONAN_FRAMEWORKS_LIBVPX )
set(CONAN_FRAMEWORKS_FOUND_LIBVPX "")  # Will be filled later
set(CONAN_DEFINES_LIBVPX )
set(CONAN_BUILD_MODULES_PATHS_LIBVPX )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBVPX )

set(CONAN_C_FLAGS_LIBVPX "")
set(CONAN_CXX_FLAGS_LIBVPX "")
set(CONAN_SHARED_LINKER_FLAGS_LIBVPX "")
set(CONAN_EXE_LINKER_FLAGS_LIBVPX "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBVPX_LIST "")
set(CONAN_CXX_FLAGS_LIBVPX_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBVPX_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBVPX_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBVPX "${CONAN_FRAMEWORKS_LIBVPX}" "_LIBVPX" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBVPX ${CONAN_PKG_LIBS_LIBVPX} ${CONAN_SYSTEM_LIBS_LIBVPX} ${CONAN_FRAMEWORKS_FOUND_LIBVPX})


#################
###  LIBMP3LAME
#################
set(CONAN_LIBMP3LAME_ROOT "/home/simon/.conan/data/libmp3lame/3.100/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_LIBMP3LAME "/home/simon/.conan/data/libmp3lame/3.100/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(CONAN_LIB_DIRS_LIBMP3LAME "/home/simon/.conan/data/libmp3lame/3.100/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_LIBMP3LAME )
set(CONAN_RES_DIRS_LIBMP3LAME )
set(CONAN_SRC_DIRS_LIBMP3LAME )
set(CONAN_BUILD_DIRS_LIBMP3LAME "/home/simon/.conan/data/libmp3lame/3.100/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/")
set(CONAN_FRAMEWORK_DIRS_LIBMP3LAME )
set(CONAN_LIBS_LIBMP3LAME mp3lame)
set(CONAN_PKG_LIBS_LIBMP3LAME mp3lame)
set(CONAN_SYSTEM_LIBS_LIBMP3LAME m)
set(CONAN_FRAMEWORKS_LIBMP3LAME )
set(CONAN_FRAMEWORKS_FOUND_LIBMP3LAME "")  # Will be filled later
set(CONAN_DEFINES_LIBMP3LAME )
set(CONAN_BUILD_MODULES_PATHS_LIBMP3LAME )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBMP3LAME )

set(CONAN_C_FLAGS_LIBMP3LAME "")
set(CONAN_CXX_FLAGS_LIBMP3LAME "")
set(CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME "")
set(CONAN_EXE_LINKER_FLAGS_LIBMP3LAME "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBMP3LAME_LIST "")
set(CONAN_CXX_FLAGS_LIBMP3LAME_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBMP3LAME_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBMP3LAME "${CONAN_FRAMEWORKS_LIBMP3LAME}" "_LIBMP3LAME" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBMP3LAME ${CONAN_PKG_LIBS_LIBMP3LAME} ${CONAN_SYSTEM_LIBS_LIBMP3LAME} ${CONAN_FRAMEWORKS_FOUND_LIBMP3LAME})


#################
###  LIBFDK_AAC
#################
set(CONAN_LIBFDK_AAC_ROOT "/home/simon/.conan/data/libfdk_aac/2.0.2/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56")
set(CONAN_INCLUDE_DIRS_LIBFDK_AAC "/home/simon/.conan/data/libfdk_aac/2.0.2/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/include")
set(CONAN_LIB_DIRS_LIBFDK_AAC "/home/simon/.conan/data/libfdk_aac/2.0.2/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/lib")
set(CONAN_BIN_DIRS_LIBFDK_AAC )
set(CONAN_RES_DIRS_LIBFDK_AAC )
set(CONAN_SRC_DIRS_LIBFDK_AAC )
set(CONAN_BUILD_DIRS_LIBFDK_AAC "/home/simon/.conan/data/libfdk_aac/2.0.2/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/")
set(CONAN_FRAMEWORK_DIRS_LIBFDK_AAC )
set(CONAN_LIBS_LIBFDK_AAC fdk-aac)
set(CONAN_PKG_LIBS_LIBFDK_AAC fdk-aac)
set(CONAN_SYSTEM_LIBS_LIBFDK_AAC m)
set(CONAN_FRAMEWORKS_LIBFDK_AAC )
set(CONAN_FRAMEWORKS_FOUND_LIBFDK_AAC "")  # Will be filled later
set(CONAN_DEFINES_LIBFDK_AAC )
set(CONAN_BUILD_MODULES_PATHS_LIBFDK_AAC )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBFDK_AAC )

set(CONAN_C_FLAGS_LIBFDK_AAC "")
set(CONAN_CXX_FLAGS_LIBFDK_AAC "")
set(CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC "")
set(CONAN_EXE_LINKER_FLAGS_LIBFDK_AAC "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBFDK_AAC_LIST "")
set(CONAN_CXX_FLAGS_LIBFDK_AAC_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBFDK_AAC_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBFDK_AAC "${CONAN_FRAMEWORKS_LIBFDK_AAC}" "_LIBFDK_AAC" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBFDK_AAC ${CONAN_PKG_LIBS_LIBFDK_AAC} ${CONAN_SYSTEM_LIBS_LIBFDK_AAC} ${CONAN_FRAMEWORKS_FOUND_LIBFDK_AAC})


#################
###  PULSEAUDIO
#################
set(CONAN_PULSEAUDIO_ROOT "/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae")
set(CONAN_INCLUDE_DIRS_PULSEAUDIO "/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae/include")
set(CONAN_LIB_DIRS_PULSEAUDIO "/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae/lib"
			"/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae/lib/pulseaudio")
set(CONAN_BIN_DIRS_PULSEAUDIO "/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae/bin")
set(CONAN_RES_DIRS_PULSEAUDIO )
set(CONAN_SRC_DIRS_PULSEAUDIO )
set(CONAN_BUILD_DIRS_PULSEAUDIO "/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae/")
set(CONAN_FRAMEWORK_DIRS_PULSEAUDIO )
set(CONAN_LIBS_PULSEAUDIO pulse-simple pulse pulsecommon-14.2)
set(CONAN_PKG_LIBS_PULSEAUDIO pulse-simple pulse pulsecommon-14.2)
set(CONAN_SYSTEM_LIBS_PULSEAUDIO )
set(CONAN_FRAMEWORKS_PULSEAUDIO )
set(CONAN_FRAMEWORKS_FOUND_PULSEAUDIO "")  # Will be filled later
set(CONAN_DEFINES_PULSEAUDIO "-D_REENTRANT")
set(CONAN_BUILD_MODULES_PATHS_PULSEAUDIO )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_PULSEAUDIO "_REENTRANT")

set(CONAN_C_FLAGS_PULSEAUDIO "")
set(CONAN_CXX_FLAGS_PULSEAUDIO "")
set(CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO "")
set(CONAN_EXE_LINKER_FLAGS_PULSEAUDIO "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_PULSEAUDIO_LIST "")
set(CONAN_CXX_FLAGS_PULSEAUDIO_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_LIST "")
set(CONAN_EXE_LINKER_FLAGS_PULSEAUDIO_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_PULSEAUDIO "${CONAN_FRAMEWORKS_PULSEAUDIO}" "_PULSEAUDIO" "")
# Append to aggregated values variable
set(CONAN_LIBS_PULSEAUDIO ${CONAN_PKG_LIBS_PULSEAUDIO} ${CONAN_SYSTEM_LIBS_PULSEAUDIO} ${CONAN_FRAMEWORKS_FOUND_PULSEAUDIO})


#################
###  VAAPI
#################
set(CONAN_VAAPI_ROOT "/home/simon/.conan/data/vaapi/system/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_VAAPI )
set(CONAN_LIB_DIRS_VAAPI )
set(CONAN_BIN_DIRS_VAAPI )
set(CONAN_RES_DIRS_VAAPI )
set(CONAN_SRC_DIRS_VAAPI )
set(CONAN_BUILD_DIRS_VAAPI "/home/simon/.conan/data/vaapi/system/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/")
set(CONAN_FRAMEWORK_DIRS_VAAPI )
set(CONAN_LIBS_VAAPI )
set(CONAN_PKG_LIBS_VAAPI )
set(CONAN_SYSTEM_LIBS_VAAPI va va-x11 va va-drm va)
set(CONAN_FRAMEWORKS_VAAPI )
set(CONAN_FRAMEWORKS_FOUND_VAAPI "")  # Will be filled later
set(CONAN_DEFINES_VAAPI )
set(CONAN_BUILD_MODULES_PATHS_VAAPI )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_VAAPI )

set(CONAN_C_FLAGS_VAAPI "")
set(CONAN_CXX_FLAGS_VAAPI "")
set(CONAN_SHARED_LINKER_FLAGS_VAAPI "")
set(CONAN_EXE_LINKER_FLAGS_VAAPI "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_VAAPI_LIST "")
set(CONAN_CXX_FLAGS_VAAPI_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_VAAPI_LIST "")
set(CONAN_EXE_LINKER_FLAGS_VAAPI_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_VAAPI "${CONAN_FRAMEWORKS_VAAPI}" "_VAAPI" "")
# Append to aggregated values variable
set(CONAN_LIBS_VAAPI ${CONAN_PKG_LIBS_VAAPI} ${CONAN_SYSTEM_LIBS_VAAPI} ${CONAN_FRAMEWORKS_FOUND_VAAPI})


#################
###  VDPAU
#################
set(CONAN_VDPAU_ROOT "/home/simon/.conan/data/vdpau/system/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_VDPAU )
set(CONAN_LIB_DIRS_VDPAU )
set(CONAN_BIN_DIRS_VDPAU )
set(CONAN_RES_DIRS_VDPAU )
set(CONAN_SRC_DIRS_VDPAU )
set(CONAN_BUILD_DIRS_VDPAU "/home/simon/.conan/data/vdpau/system/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/")
set(CONAN_FRAMEWORK_DIRS_VDPAU )
set(CONAN_LIBS_VDPAU )
set(CONAN_PKG_LIBS_VDPAU )
set(CONAN_SYSTEM_LIBS_VDPAU vdpau)
set(CONAN_FRAMEWORKS_VDPAU )
set(CONAN_FRAMEWORKS_FOUND_VDPAU "")  # Will be filled later
set(CONAN_DEFINES_VDPAU )
set(CONAN_BUILD_MODULES_PATHS_VDPAU )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_VDPAU )

set(CONAN_C_FLAGS_VDPAU "")
set(CONAN_CXX_FLAGS_VDPAU "")
set(CONAN_SHARED_LINKER_FLAGS_VDPAU "")
set(CONAN_EXE_LINKER_FLAGS_VDPAU "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_VDPAU_LIST "")
set(CONAN_CXX_FLAGS_VDPAU_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_VDPAU_LIST "")
set(CONAN_EXE_LINKER_FLAGS_VDPAU_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_VDPAU "${CONAN_FRAMEWORKS_VDPAU}" "_VDPAU" "")
# Append to aggregated values variable
set(CONAN_LIBS_VDPAU ${CONAN_PKG_LIBS_VDPAU} ${CONAN_SYSTEM_LIBS_VDPAU} ${CONAN_FRAMEWORKS_FOUND_VDPAU})


#################
###  LIBPNG
#################
set(CONAN_LIBPNG_ROOT "/home/simon/.conan/data/libpng/1.6.37/_/_/package/7929d8ecf29c60d74fd3c1f6cb78bbb3cb49c0c7")
set(CONAN_INCLUDE_DIRS_LIBPNG "/home/simon/.conan/data/libpng/1.6.37/_/_/package/7929d8ecf29c60d74fd3c1f6cb78bbb3cb49c0c7/include")
set(CONAN_LIB_DIRS_LIBPNG "/home/simon/.conan/data/libpng/1.6.37/_/_/package/7929d8ecf29c60d74fd3c1f6cb78bbb3cb49c0c7/lib")
set(CONAN_BIN_DIRS_LIBPNG )
set(CONAN_RES_DIRS_LIBPNG )
set(CONAN_SRC_DIRS_LIBPNG )
set(CONAN_BUILD_DIRS_LIBPNG "/home/simon/.conan/data/libpng/1.6.37/_/_/package/7929d8ecf29c60d74fd3c1f6cb78bbb3cb49c0c7/")
set(CONAN_FRAMEWORK_DIRS_LIBPNG )
set(CONAN_LIBS_LIBPNG png16)
set(CONAN_PKG_LIBS_LIBPNG png16)
set(CONAN_SYSTEM_LIBS_LIBPNG m)
set(CONAN_FRAMEWORKS_LIBPNG )
set(CONAN_FRAMEWORKS_FOUND_LIBPNG "")  # Will be filled later
set(CONAN_DEFINES_LIBPNG )
set(CONAN_BUILD_MODULES_PATHS_LIBPNG )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBPNG )

set(CONAN_C_FLAGS_LIBPNG "")
set(CONAN_CXX_FLAGS_LIBPNG "")
set(CONAN_SHARED_LINKER_FLAGS_LIBPNG "")
set(CONAN_EXE_LINKER_FLAGS_LIBPNG "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBPNG_LIST "")
set(CONAN_CXX_FLAGS_LIBPNG_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBPNG_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBPNG_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBPNG "${CONAN_FRAMEWORKS_LIBPNG}" "_LIBPNG" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBPNG ${CONAN_PKG_LIBS_LIBPNG} ${CONAN_SYSTEM_LIBS_LIBPNG} ${CONAN_FRAMEWORKS_FOUND_LIBPNG})


#################
###  BZIP2
#################
set(CONAN_BZIP2_ROOT "/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0")
set(CONAN_INCLUDE_DIRS_BZIP2 "/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0/include")
set(CONAN_LIB_DIRS_BZIP2 "/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0/lib")
set(CONAN_BIN_DIRS_BZIP2 "/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0/bin")
set(CONAN_RES_DIRS_BZIP2 )
set(CONAN_SRC_DIRS_BZIP2 )
set(CONAN_BUILD_DIRS_BZIP2 "/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0/"
			"/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_BZIP2 )
set(CONAN_LIBS_BZIP2 bz2)
set(CONAN_PKG_LIBS_BZIP2 bz2)
set(CONAN_SYSTEM_LIBS_BZIP2 )
set(CONAN_FRAMEWORKS_BZIP2 )
set(CONAN_FRAMEWORKS_FOUND_BZIP2 "")  # Will be filled later
set(CONAN_DEFINES_BZIP2 )
set(CONAN_BUILD_MODULES_PATHS_BZIP2 )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_BZIP2 )

set(CONAN_C_FLAGS_BZIP2 "")
set(CONAN_CXX_FLAGS_BZIP2 "")
set(CONAN_SHARED_LINKER_FLAGS_BZIP2 "")
set(CONAN_EXE_LINKER_FLAGS_BZIP2 "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_BZIP2_LIST "")
set(CONAN_CXX_FLAGS_BZIP2_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_BZIP2_LIST "")
set(CONAN_EXE_LINKER_FLAGS_BZIP2_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_BZIP2 "${CONAN_FRAMEWORKS_BZIP2}" "_BZIP2" "")
# Append to aggregated values variable
set(CONAN_LIBS_BZIP2 ${CONAN_PKG_LIBS_BZIP2} ${CONAN_SYSTEM_LIBS_BZIP2} ${CONAN_FRAMEWORKS_FOUND_BZIP2})


#################
###  LIBICONV
#################
set(CONAN_LIBICONV_ROOT "/home/simon/.conan/data/libiconv/1.16/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_LIBICONV "/home/simon/.conan/data/libiconv/1.16/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(CONAN_LIB_DIRS_LIBICONV "/home/simon/.conan/data/libiconv/1.16/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_LIBICONV "/home/simon/.conan/data/libiconv/1.16/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/bin")
set(CONAN_RES_DIRS_LIBICONV )
set(CONAN_SRC_DIRS_LIBICONV )
set(CONAN_BUILD_DIRS_LIBICONV "/home/simon/.conan/data/libiconv/1.16/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/")
set(CONAN_FRAMEWORK_DIRS_LIBICONV )
set(CONAN_LIBS_LIBICONV iconv charset)
set(CONAN_PKG_LIBS_LIBICONV iconv charset)
set(CONAN_SYSTEM_LIBS_LIBICONV )
set(CONAN_FRAMEWORKS_LIBICONV )
set(CONAN_FRAMEWORKS_FOUND_LIBICONV "")  # Will be filled later
set(CONAN_DEFINES_LIBICONV )
set(CONAN_BUILD_MODULES_PATHS_LIBICONV )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBICONV )

set(CONAN_C_FLAGS_LIBICONV "")
set(CONAN_CXX_FLAGS_LIBICONV "")
set(CONAN_SHARED_LINKER_FLAGS_LIBICONV "")
set(CONAN_EXE_LINKER_FLAGS_LIBICONV "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBICONV_LIST "")
set(CONAN_CXX_FLAGS_LIBICONV_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBICONV_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBICONV_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBICONV "${CONAN_FRAMEWORKS_LIBICONV}" "_LIBICONV" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBICONV ${CONAN_PKG_LIBS_LIBICONV} ${CONAN_SYSTEM_LIBS_LIBICONV} ${CONAN_FRAMEWORKS_FOUND_LIBICONV})


#################
###  BROTLI
#################
set(CONAN_BROTLI_ROOT "/home/simon/.conan/data/brotli/1.0.9/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_BROTLI "/home/simon/.conan/data/brotli/1.0.9/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/brotli/1.0.9/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include/brotli")
set(CONAN_LIB_DIRS_BROTLI "/home/simon/.conan/data/brotli/1.0.9/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_BROTLI )
set(CONAN_RES_DIRS_BROTLI )
set(CONAN_SRC_DIRS_BROTLI )
set(CONAN_BUILD_DIRS_BROTLI "/home/simon/.conan/data/brotli/1.0.9/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/")
set(CONAN_FRAMEWORK_DIRS_BROTLI )
set(CONAN_LIBS_BROTLI brotlidec-static brotlienc-static brotlicommon-static)
set(CONAN_PKG_LIBS_BROTLI brotlidec-static brotlienc-static brotlicommon-static)
set(CONAN_SYSTEM_LIBS_BROTLI m)
set(CONAN_FRAMEWORKS_BROTLI )
set(CONAN_FRAMEWORKS_FOUND_BROTLI "")  # Will be filled later
set(CONAN_DEFINES_BROTLI )
set(CONAN_BUILD_MODULES_PATHS_BROTLI )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_BROTLI )

set(CONAN_C_FLAGS_BROTLI "")
set(CONAN_CXX_FLAGS_BROTLI "")
set(CONAN_SHARED_LINKER_FLAGS_BROTLI "")
set(CONAN_EXE_LINKER_FLAGS_BROTLI "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_BROTLI_LIST "")
set(CONAN_CXX_FLAGS_BROTLI_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_BROTLI_LIST "")
set(CONAN_EXE_LINKER_FLAGS_BROTLI_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_BROTLI "${CONAN_FRAMEWORKS_BROTLI}" "_BROTLI" "")
# Append to aggregated values variable
set(CONAN_LIBS_BROTLI ${CONAN_PKG_LIBS_BROTLI} ${CONAN_SYSTEM_LIBS_BROTLI} ${CONAN_FRAMEWORKS_FOUND_BROTLI})


#################
###  OPENSSL
#################
set(CONAN_OPENSSL_ROOT "/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_OPENSSL "/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(CONAN_LIB_DIRS_OPENSSL "/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_OPENSSL "/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/bin")
set(CONAN_RES_DIRS_OPENSSL )
set(CONAN_SRC_DIRS_OPENSSL )
set(CONAN_BUILD_DIRS_OPENSSL "/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib/cmake")
set(CONAN_FRAMEWORK_DIRS_OPENSSL )
set(CONAN_LIBS_OPENSSL ssl crypto)
set(CONAN_PKG_LIBS_OPENSSL ssl crypto)
set(CONAN_SYSTEM_LIBS_OPENSSL dl pthread rt)
set(CONAN_FRAMEWORKS_OPENSSL )
set(CONAN_FRAMEWORKS_FOUND_OPENSSL "")  # Will be filled later
set(CONAN_DEFINES_OPENSSL )
set(CONAN_BUILD_MODULES_PATHS_OPENSSL )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_OPENSSL )

set(CONAN_C_FLAGS_OPENSSL "")
set(CONAN_CXX_FLAGS_OPENSSL "")
set(CONAN_SHARED_LINKER_FLAGS_OPENSSL "")
set(CONAN_EXE_LINKER_FLAGS_OPENSSL "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_OPENSSL_LIST "")
set(CONAN_CXX_FLAGS_OPENSSL_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_OPENSSL_LIST "")
set(CONAN_EXE_LINKER_FLAGS_OPENSSL_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_OPENSSL "${CONAN_FRAMEWORKS_OPENSSL}" "_OPENSSL" "")
# Append to aggregated values variable
set(CONAN_LIBS_OPENSSL ${CONAN_PKG_LIBS_OPENSSL} ${CONAN_SYSTEM_LIBS_OPENSSL} ${CONAN_FRAMEWORKS_FOUND_OPENSSL})


#################
###  LIBALSA
#################
set(CONAN_LIBALSA_ROOT "/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d")
set(CONAN_INCLUDE_DIRS_LIBALSA "/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d/include")
set(CONAN_LIB_DIRS_LIBALSA "/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d/lib")
set(CONAN_BIN_DIRS_LIBALSA "/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d/bin")
set(CONAN_RES_DIRS_LIBALSA "/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d/res")
set(CONAN_SRC_DIRS_LIBALSA )
set(CONAN_BUILD_DIRS_LIBALSA "/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d/")
set(CONAN_FRAMEWORK_DIRS_LIBALSA )
set(CONAN_LIBS_LIBALSA asound)
set(CONAN_PKG_LIBS_LIBALSA asound)
set(CONAN_SYSTEM_LIBS_LIBALSA dl m rt pthread)
set(CONAN_FRAMEWORKS_LIBALSA )
set(CONAN_FRAMEWORKS_FOUND_LIBALSA "")  # Will be filled later
set(CONAN_DEFINES_LIBALSA )
set(CONAN_BUILD_MODULES_PATHS_LIBALSA )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBALSA )

set(CONAN_C_FLAGS_LIBALSA "")
set(CONAN_CXX_FLAGS_LIBALSA "")
set(CONAN_SHARED_LINKER_FLAGS_LIBALSA "")
set(CONAN_EXE_LINKER_FLAGS_LIBALSA "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBALSA_LIST "")
set(CONAN_CXX_FLAGS_LIBALSA_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBALSA_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBALSA_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBALSA "${CONAN_FRAMEWORKS_LIBALSA}" "_LIBALSA" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBALSA ${CONAN_PKG_LIBS_LIBALSA} ${CONAN_SYSTEM_LIBS_LIBALSA} ${CONAN_FRAMEWORKS_FOUND_LIBALSA})


#################
###  XORG
#################
set(CONAN_XORG_ROOT "/home/simon/.conan/data/xorg/system/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(CONAN_INCLUDE_DIRS_XORG "/usr/include/uuid"
			"/usr/include/freetype2"
			"/usr/include/libpng16")
set(CONAN_LIB_DIRS_XORG )
set(CONAN_BIN_DIRS_XORG )
set(CONAN_RES_DIRS_XORG )
set(CONAN_SRC_DIRS_XORG )
set(CONAN_BUILD_DIRS_XORG "/home/simon/.conan/data/xorg/system/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/")
set(CONAN_FRAMEWORK_DIRS_XORG )
set(CONAN_LIBS_XORG )
set(CONAN_PKG_LIBS_XORG )
set(CONAN_SYSTEM_LIBS_XORG X11 X11-xcb xcb fontenc ICE SM Xau Xaw7 Xt Xcomposite Xcursor Xdamage Xfixes Xdmcp Xext Xft Xi Xinerama xkbfile Xmu Xmuu Xpm Xrandr Xrender XRes Xss Xtst Xv XvMC Xxf86vm xcb-xkb xcb-icccm xcb-image xcb-shm xcb-keysyms xcb-randr xcb-render xcb-render-util xcb-shape xcb-sync xcb-xfixes xcb-xinerama xcb-util xcb-dri3)
set(CONAN_FRAMEWORKS_XORG )
set(CONAN_FRAMEWORKS_FOUND_XORG "")  # Will be filled later
set(CONAN_DEFINES_XORG "-D_DEFAULT_SOURCE"
			"-D_BSD_SOURCE"
			"-DHAS_FCHOWN"
			"-DHAS_STICKY_DIR_BIT")
set(CONAN_BUILD_MODULES_PATHS_XORG )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_XORG "_DEFAULT_SOURCE"
			"_BSD_SOURCE"
			"HAS_FCHOWN"
			"HAS_STICKY_DIR_BIT")

set(CONAN_C_FLAGS_XORG "")
set(CONAN_CXX_FLAGS_XORG "")
set(CONAN_SHARED_LINKER_FLAGS_XORG "")
set(CONAN_EXE_LINKER_FLAGS_XORG "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_XORG_LIST "")
set(CONAN_CXX_FLAGS_XORG_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_XORG_LIST "")
set(CONAN_EXE_LINKER_FLAGS_XORG_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_XORG "${CONAN_FRAMEWORKS_XORG}" "_XORG" "")
# Append to aggregated values variable
set(CONAN_LIBS_XORG ${CONAN_PKG_LIBS_XORG} ${CONAN_SYSTEM_LIBS_XORG} ${CONAN_FRAMEWORKS_FOUND_XORG})


#################
###  LIBSNDFILE
#################
set(CONAN_LIBSNDFILE_ROOT "/home/simon/.conan/data/libsndfile/1.0.31/_/_/package/9384eaf0a0891af0df136cb9739770bd5d841e22")
set(CONAN_INCLUDE_DIRS_LIBSNDFILE "/home/simon/.conan/data/libsndfile/1.0.31/_/_/package/9384eaf0a0891af0df136cb9739770bd5d841e22/include")
set(CONAN_LIB_DIRS_LIBSNDFILE "/home/simon/.conan/data/libsndfile/1.0.31/_/_/package/9384eaf0a0891af0df136cb9739770bd5d841e22/lib")
set(CONAN_BIN_DIRS_LIBSNDFILE "/home/simon/.conan/data/libsndfile/1.0.31/_/_/package/9384eaf0a0891af0df136cb9739770bd5d841e22/bin")
set(CONAN_RES_DIRS_LIBSNDFILE )
set(CONAN_SRC_DIRS_LIBSNDFILE )
set(CONAN_BUILD_DIRS_LIBSNDFILE "/home/simon/.conan/data/libsndfile/1.0.31/_/_/package/9384eaf0a0891af0df136cb9739770bd5d841e22/")
set(CONAN_FRAMEWORK_DIRS_LIBSNDFILE )
set(CONAN_LIBS_LIBSNDFILE sndfile)
set(CONAN_PKG_LIBS_LIBSNDFILE sndfile)
set(CONAN_SYSTEM_LIBS_LIBSNDFILE m dl pthread rt)
set(CONAN_FRAMEWORKS_LIBSNDFILE )
set(CONAN_FRAMEWORKS_FOUND_LIBSNDFILE "")  # Will be filled later
set(CONAN_DEFINES_LIBSNDFILE )
set(CONAN_BUILD_MODULES_PATHS_LIBSNDFILE )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBSNDFILE )

set(CONAN_C_FLAGS_LIBSNDFILE "")
set(CONAN_CXX_FLAGS_LIBSNDFILE "")
set(CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE "")
set(CONAN_EXE_LINKER_FLAGS_LIBSNDFILE "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBSNDFILE_LIST "")
set(CONAN_CXX_FLAGS_LIBSNDFILE_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBSNDFILE_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBSNDFILE "${CONAN_FRAMEWORKS_LIBSNDFILE}" "_LIBSNDFILE" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBSNDFILE ${CONAN_PKG_LIBS_LIBSNDFILE} ${CONAN_SYSTEM_LIBS_LIBSNDFILE} ${CONAN_FRAMEWORKS_FOUND_LIBSNDFILE})


#################
###  LIBCAP
#################
set(CONAN_LIBCAP_ROOT "/home/simon/.conan/data/libcap/2.50/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_LIBCAP "/home/simon/.conan/data/libcap/2.50/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(CONAN_LIB_DIRS_LIBCAP "/home/simon/.conan/data/libcap/2.50/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_LIBCAP )
set(CONAN_RES_DIRS_LIBCAP )
set(CONAN_SRC_DIRS_LIBCAP )
set(CONAN_BUILD_DIRS_LIBCAP "/home/simon/.conan/data/libcap/2.50/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/")
set(CONAN_FRAMEWORK_DIRS_LIBCAP )
set(CONAN_LIBS_LIBCAP cap)
set(CONAN_PKG_LIBS_LIBCAP cap)
set(CONAN_SYSTEM_LIBS_LIBCAP )
set(CONAN_FRAMEWORKS_LIBCAP )
set(CONAN_FRAMEWORKS_FOUND_LIBCAP "")  # Will be filled later
set(CONAN_DEFINES_LIBCAP )
set(CONAN_BUILD_MODULES_PATHS_LIBCAP )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_LIBCAP )

set(CONAN_C_FLAGS_LIBCAP "")
set(CONAN_CXX_FLAGS_LIBCAP "")
set(CONAN_SHARED_LINKER_FLAGS_LIBCAP "")
set(CONAN_EXE_LINKER_FLAGS_LIBCAP "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_LIBCAP_LIST "")
set(CONAN_CXX_FLAGS_LIBCAP_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_LIBCAP_LIST "")
set(CONAN_EXE_LINKER_FLAGS_LIBCAP_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_LIBCAP "${CONAN_FRAMEWORKS_LIBCAP}" "_LIBCAP" "")
# Append to aggregated values variable
set(CONAN_LIBS_LIBCAP ${CONAN_PKG_LIBS_LIBCAP} ${CONAN_SYSTEM_LIBS_LIBCAP} ${CONAN_FRAMEWORKS_FOUND_LIBCAP})


#################
###  ZLIB
#################
set(CONAN_ZLIB_ROOT "/home/simon/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_ZLIB "/home/simon/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(CONAN_LIB_DIRS_ZLIB "/home/simon/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_ZLIB )
set(CONAN_RES_DIRS_ZLIB )
set(CONAN_SRC_DIRS_ZLIB )
set(CONAN_BUILD_DIRS_ZLIB "/home/simon/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/")
set(CONAN_FRAMEWORK_DIRS_ZLIB )
set(CONAN_LIBS_ZLIB z)
set(CONAN_PKG_LIBS_ZLIB z)
set(CONAN_SYSTEM_LIBS_ZLIB )
set(CONAN_FRAMEWORKS_ZLIB )
set(CONAN_FRAMEWORKS_FOUND_ZLIB "")  # Will be filled later
set(CONAN_DEFINES_ZLIB )
set(CONAN_BUILD_MODULES_PATHS_ZLIB )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_ZLIB )

set(CONAN_C_FLAGS_ZLIB "")
set(CONAN_CXX_FLAGS_ZLIB "")
set(CONAN_SHARED_LINKER_FLAGS_ZLIB "")
set(CONAN_EXE_LINKER_FLAGS_ZLIB "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_ZLIB_LIST "")
set(CONAN_CXX_FLAGS_ZLIB_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_ZLIB_LIST "")
set(CONAN_EXE_LINKER_FLAGS_ZLIB_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_ZLIB "${CONAN_FRAMEWORKS_ZLIB}" "_ZLIB" "")
# Append to aggregated values variable
set(CONAN_LIBS_ZLIB ${CONAN_PKG_LIBS_ZLIB} ${CONAN_SYSTEM_LIBS_ZLIB} ${CONAN_FRAMEWORKS_FOUND_ZLIB})


#################
###  VORBIS
#################
set(CONAN_VORBIS_ROOT "/home/simon/.conan/data/vorbis/1.3.7/_/_/package/033a9a186751ed382a6fa5fa2c844ad34b7c3302")
set(CONAN_INCLUDE_DIRS_VORBIS "/home/simon/.conan/data/vorbis/1.3.7/_/_/package/033a9a186751ed382a6fa5fa2c844ad34b7c3302/include")
set(CONAN_LIB_DIRS_VORBIS "/home/simon/.conan/data/vorbis/1.3.7/_/_/package/033a9a186751ed382a6fa5fa2c844ad34b7c3302/lib")
set(CONAN_BIN_DIRS_VORBIS )
set(CONAN_RES_DIRS_VORBIS )
set(CONAN_SRC_DIRS_VORBIS )
set(CONAN_BUILD_DIRS_VORBIS "/home/simon/.conan/data/vorbis/1.3.7/_/_/package/033a9a186751ed382a6fa5fa2c844ad34b7c3302/")
set(CONAN_FRAMEWORK_DIRS_VORBIS )
set(CONAN_LIBS_VORBIS vorbisenc vorbisfile vorbis)
set(CONAN_PKG_LIBS_VORBIS vorbisenc vorbisfile vorbis)
set(CONAN_SYSTEM_LIBS_VORBIS m)
set(CONAN_FRAMEWORKS_VORBIS )
set(CONAN_FRAMEWORKS_FOUND_VORBIS "")  # Will be filled later
set(CONAN_DEFINES_VORBIS )
set(CONAN_BUILD_MODULES_PATHS_VORBIS )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_VORBIS )

set(CONAN_C_FLAGS_VORBIS "")
set(CONAN_CXX_FLAGS_VORBIS "")
set(CONAN_SHARED_LINKER_FLAGS_VORBIS "")
set(CONAN_EXE_LINKER_FLAGS_VORBIS "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_VORBIS_LIST "")
set(CONAN_CXX_FLAGS_VORBIS_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_VORBIS_LIST "")
set(CONAN_EXE_LINKER_FLAGS_VORBIS_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_VORBIS "${CONAN_FRAMEWORKS_VORBIS}" "_VORBIS" "")
# Append to aggregated values variable
set(CONAN_LIBS_VORBIS ${CONAN_PKG_LIBS_VORBIS} ${CONAN_SYSTEM_LIBS_VORBIS} ${CONAN_FRAMEWORKS_FOUND_VORBIS})


#################
###  OPUS
#################
set(CONAN_OPUS_ROOT "/home/simon/.conan/data/opus/1.3.1/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_OPUS "/home/simon/.conan/data/opus/1.3.1/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/opus/1.3.1/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include/opus")
set(CONAN_LIB_DIRS_OPUS "/home/simon/.conan/data/opus/1.3.1/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_OPUS )
set(CONAN_RES_DIRS_OPUS )
set(CONAN_SRC_DIRS_OPUS )
set(CONAN_BUILD_DIRS_OPUS "/home/simon/.conan/data/opus/1.3.1/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/")
set(CONAN_FRAMEWORK_DIRS_OPUS )
set(CONAN_LIBS_OPUS opus)
set(CONAN_PKG_LIBS_OPUS opus)
set(CONAN_SYSTEM_LIBS_OPUS m)
set(CONAN_FRAMEWORKS_OPUS )
set(CONAN_FRAMEWORKS_FOUND_OPUS "")  # Will be filled later
set(CONAN_DEFINES_OPUS )
set(CONAN_BUILD_MODULES_PATHS_OPUS )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_OPUS )

set(CONAN_C_FLAGS_OPUS "")
set(CONAN_CXX_FLAGS_OPUS "")
set(CONAN_SHARED_LINKER_FLAGS_OPUS "")
set(CONAN_EXE_LINKER_FLAGS_OPUS "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_OPUS_LIST "")
set(CONAN_CXX_FLAGS_OPUS_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_OPUS_LIST "")
set(CONAN_EXE_LINKER_FLAGS_OPUS_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_OPUS "${CONAN_FRAMEWORKS_OPUS}" "_OPUS" "")
# Append to aggregated values variable
set(CONAN_LIBS_OPUS ${CONAN_PKG_LIBS_OPUS} ${CONAN_SYSTEM_LIBS_OPUS} ${CONAN_FRAMEWORKS_FOUND_OPUS})


#################
###  FLAC
#################
set(CONAN_FLAC_ROOT "/home/simon/.conan/data/flac/1.3.3/_/_/package/b9bd28193f63cd74fd2213b63c74017e7893c24d")
set(CONAN_INCLUDE_DIRS_FLAC "/home/simon/.conan/data/flac/1.3.3/_/_/package/b9bd28193f63cd74fd2213b63c74017e7893c24d/include")
set(CONAN_LIB_DIRS_FLAC "/home/simon/.conan/data/flac/1.3.3/_/_/package/b9bd28193f63cd74fd2213b63c74017e7893c24d/lib")
set(CONAN_BIN_DIRS_FLAC "/home/simon/.conan/data/flac/1.3.3/_/_/package/b9bd28193f63cd74fd2213b63c74017e7893c24d/bin")
set(CONAN_RES_DIRS_FLAC )
set(CONAN_SRC_DIRS_FLAC )
set(CONAN_BUILD_DIRS_FLAC "/home/simon/.conan/data/flac/1.3.3/_/_/package/b9bd28193f63cd74fd2213b63c74017e7893c24d/")
set(CONAN_FRAMEWORK_DIRS_FLAC )
set(CONAN_LIBS_FLAC FLAC++ FLAC)
set(CONAN_PKG_LIBS_FLAC FLAC++ FLAC)
set(CONAN_SYSTEM_LIBS_FLAC m)
set(CONAN_FRAMEWORKS_FLAC )
set(CONAN_FRAMEWORKS_FOUND_FLAC "")  # Will be filled later
set(CONAN_DEFINES_FLAC "-DFLAC__NO_DLL")
set(CONAN_BUILD_MODULES_PATHS_FLAC )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_FLAC "FLAC__NO_DLL")

set(CONAN_C_FLAGS_FLAC "")
set(CONAN_CXX_FLAGS_FLAC "")
set(CONAN_SHARED_LINKER_FLAGS_FLAC "")
set(CONAN_EXE_LINKER_FLAGS_FLAC "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_FLAC_LIST "")
set(CONAN_CXX_FLAGS_FLAC_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_FLAC_LIST "")
set(CONAN_EXE_LINKER_FLAGS_FLAC_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_FLAC "${CONAN_FRAMEWORKS_FLAC}" "_FLAC" "")
# Append to aggregated values variable
set(CONAN_LIBS_FLAC ${CONAN_PKG_LIBS_FLAC} ${CONAN_SYSTEM_LIBS_FLAC} ${CONAN_FRAMEWORKS_FOUND_FLAC})


#################
###  OGG
#################
set(CONAN_OGG_ROOT "/home/simon/.conan/data/ogg/1.3.4/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709")
set(CONAN_INCLUDE_DIRS_OGG "/home/simon/.conan/data/ogg/1.3.4/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(CONAN_LIB_DIRS_OGG "/home/simon/.conan/data/ogg/1.3.4/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(CONAN_BIN_DIRS_OGG )
set(CONAN_RES_DIRS_OGG )
set(CONAN_SRC_DIRS_OGG )
set(CONAN_BUILD_DIRS_OGG "/home/simon/.conan/data/ogg/1.3.4/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/")
set(CONAN_FRAMEWORK_DIRS_OGG )
set(CONAN_LIBS_OGG ogg)
set(CONAN_PKG_LIBS_OGG ogg)
set(CONAN_SYSTEM_LIBS_OGG )
set(CONAN_FRAMEWORKS_OGG )
set(CONAN_FRAMEWORKS_FOUND_OGG "")  # Will be filled later
set(CONAN_DEFINES_OGG )
set(CONAN_BUILD_MODULES_PATHS_OGG )
# COMPILE_DEFINITIONS are equal to CONAN_DEFINES without -D, for targets
set(CONAN_COMPILE_DEFINITIONS_OGG )

set(CONAN_C_FLAGS_OGG "")
set(CONAN_CXX_FLAGS_OGG "")
set(CONAN_SHARED_LINKER_FLAGS_OGG "")
set(CONAN_EXE_LINKER_FLAGS_OGG "")

# For modern cmake targets we use the list variables (separated with ;)
set(CONAN_C_FLAGS_OGG_LIST "")
set(CONAN_CXX_FLAGS_OGG_LIST "")
set(CONAN_SHARED_LINKER_FLAGS_OGG_LIST "")
set(CONAN_EXE_LINKER_FLAGS_OGG_LIST "")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND_OGG "${CONAN_FRAMEWORKS_OGG}" "_OGG" "")
# Append to aggregated values variable
set(CONAN_LIBS_OGG ${CONAN_PKG_LIBS_OGG} ${CONAN_SYSTEM_LIBS_OGG} ${CONAN_FRAMEWORKS_FOUND_OGG})


### Definition of global aggregated variables ###

set(CONAN_PACKAGE_NAME None)
set(CONAN_PACKAGE_VERSION None)

set(CONAN_SETTINGS_ARCH "x86_64")
set(CONAN_SETTINGS_ARCH_BUILD "x86_64")
set(CONAN_SETTINGS_BUILD_TYPE "Release")
set(CONAN_SETTINGS_COMPILER "gcc")
set(CONAN_SETTINGS_COMPILER_LIBCXX "libstdc++11")
set(CONAN_SETTINGS_COMPILER_VERSION "9")
set(CONAN_SETTINGS_OS "Linux")
set(CONAN_SETTINGS_OS_BUILD "Linux")

set(CONAN_DEPENDENCIES opencv jasper openexr libtiff eigen ffmpeg quirc gtk protobuf ade libjpeg libdeflate xz_utils jbig zstd libwebp freetype openjpeg openh264 libx264 libx265 libvpx libmp3lame libfdk_aac pulseaudio vaapi vdpau libpng bzip2 libiconv brotli openssl libalsa xorg libsndfile libcap zlib vorbis opus flac ogg)
# Storing original command line args (CMake helper) flags
set(CONAN_CMD_CXX_FLAGS ${CONAN_CXX_FLAGS})

set(CONAN_CMD_SHARED_LINKER_FLAGS ${CONAN_SHARED_LINKER_FLAGS})
set(CONAN_CMD_C_FLAGS ${CONAN_C_FLAGS})
# Defining accumulated conan variables for all deps

set(CONAN_INCLUDE_DIRS "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/include"
			"/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/include/opencv4"
			"/home/simon/.conan/data/jasper/2.0.32/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/include"
			"/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/include"
			"/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/include/OpenEXR"
			"/home/simon/.conan/data/libtiff/4.3.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18/include"
			"/home/simon/.conan/data/eigen/3.3.9/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3"
			"/home/simon/.conan/data/ffmpeg/4.4/_/_/package/2770b46b8645e6555ada38043cbb4bf19de6ac0f/include"
			"/home/simon/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711/include"
			"/usr/include/gtk-2.0"
			"/usr/lib/x86_64-linux-gnu/gtk-2.0/include"
			"/usr/include/pango-1.0"
			"/usr/include/atk-1.0"
			"/usr/include/gdk-pixbuf-2.0"
			"/usr/include/libmount"
			"/usr/include/blkid"
			"/usr/include/pango-1.0"
			"/usr/include/fribidi"
			"/usr/include/cairo"
			"/usr/include/pixman-1"
			"/usr/include/harfbuzz"
			"/usr/include/glib-2.0"
			"/usr/lib/x86_64-linux-gnu/glib-2.0/include"
			"/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/include"
			"/home/simon/.conan/data/ade/0.1.1f/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/include"
			"/home/simon/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/libdeflate/1.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/include"
			"/home/simon/.conan/data/zstd/1.5.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/libwebp/1.2.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656/include"
			"/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/include"
			"/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/include/freetype2"
			"/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include/openjpeg-2.4"
			"/home/simon/.conan/data/openh264/2.1.1/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/include"
			"/home/simon/.conan/data/libx264/20191217/_/_/package/52bfaf5780e9ee7fba3fa4a492156dbcf0dff057/include"
			"/home/simon/.conan/data/libx265/3.4/_/_/package/75e903d40ade00f9adc18d07c1d36ecb3e595a39/include"
			"/home/simon/.conan/data/libvpx/1.10.0/_/_/package/21d176c6d550e07ab7b17c1513a8ddb7fb960145/include"
			"/home/simon/.conan/data/libmp3lame/3.100/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/libfdk_aac/2.0.2/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/include"
			"/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae/include"
			"/home/simon/.conan/data/libpng/1.6.37/_/_/package/7929d8ecf29c60d74fd3c1f6cb78bbb3cb49c0c7/include"
			"/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0/include"
			"/home/simon/.conan/data/libiconv/1.16/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/brotli/1.0.9/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/brotli/1.0.9/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include/brotli"
			"/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d/include"
			"/usr/include/uuid"
			"/usr/include/freetype2"
			"/usr/include/libpng16"
			"/home/simon/.conan/data/libsndfile/1.0.31/_/_/package/9384eaf0a0891af0df136cb9739770bd5d841e22/include"
			"/home/simon/.conan/data/libcap/2.50/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/vorbis/1.3.7/_/_/package/033a9a186751ed382a6fa5fa2c844ad34b7c3302/include"
			"/home/simon/.conan/data/opus/1.3.1/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include"
			"/home/simon/.conan/data/opus/1.3.1/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include/opus"
			"/home/simon/.conan/data/flac/1.3.3/_/_/package/b9bd28193f63cd74fd2213b63c74017e7893c24d/include"
			"/home/simon/.conan/data/ogg/1.3.4/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include" ${CONAN_INCLUDE_DIRS})
set(CONAN_LIB_DIRS "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/lib"
			"/home/simon/.conan/data/jasper/2.0.32/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/lib"
			"/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/lib"
			"/home/simon/.conan/data/libtiff/4.3.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18/lib"
			"/home/simon/.conan/data/ffmpeg/4.4/_/_/package/2770b46b8645e6555ada38043cbb4bf19de6ac0f/lib"
			"/home/simon/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711/lib"
			"/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/lib"
			"/home/simon/.conan/data/ade/0.1.1f/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/lib"
			"/home/simon/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/libdeflate/1.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/lib"
			"/home/simon/.conan/data/zstd/1.5.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/libwebp/1.2.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656/lib"
			"/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/lib"
			"/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/openh264/2.1.1/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/lib"
			"/home/simon/.conan/data/libx264/20191217/_/_/package/52bfaf5780e9ee7fba3fa4a492156dbcf0dff057/lib"
			"/home/simon/.conan/data/libx265/3.4/_/_/package/75e903d40ade00f9adc18d07c1d36ecb3e595a39/lib"
			"/home/simon/.conan/data/libvpx/1.10.0/_/_/package/21d176c6d550e07ab7b17c1513a8ddb7fb960145/lib"
			"/home/simon/.conan/data/libmp3lame/3.100/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/libfdk_aac/2.0.2/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/lib"
			"/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae/lib"
			"/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae/lib/pulseaudio"
			"/home/simon/.conan/data/libpng/1.6.37/_/_/package/7929d8ecf29c60d74fd3c1f6cb78bbb3cb49c0c7/lib"
			"/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0/lib"
			"/home/simon/.conan/data/libiconv/1.16/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/brotli/1.0.9/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d/lib"
			"/home/simon/.conan/data/libsndfile/1.0.31/_/_/package/9384eaf0a0891af0df136cb9739770bd5d841e22/lib"
			"/home/simon/.conan/data/libcap/2.50/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/vorbis/1.3.7/_/_/package/033a9a186751ed382a6fa5fa2c844ad34b7c3302/lib"
			"/home/simon/.conan/data/opus/1.3.1/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib"
			"/home/simon/.conan/data/flac/1.3.3/_/_/package/b9bd28193f63cd74fd2213b63c74017e7893c24d/lib"
			"/home/simon/.conan/data/ogg/1.3.4/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib" ${CONAN_LIB_DIRS})
set(CONAN_BIN_DIRS "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/bin"
			"/home/simon/.conan/data/ffmpeg/4.4/_/_/package/2770b46b8645e6555ada38043cbb4bf19de6ac0f/bin"
			"/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/bin"
			"/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/bin"
			"/home/simon/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/bin"
			"/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/bin"
			"/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae/bin"
			"/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0/bin"
			"/home/simon/.conan/data/libiconv/1.16/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/bin"
			"/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/bin"
			"/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d/bin"
			"/home/simon/.conan/data/libsndfile/1.0.31/_/_/package/9384eaf0a0891af0df136cb9739770bd5d841e22/bin"
			"/home/simon/.conan/data/flac/1.3.3/_/_/package/b9bd28193f63cd74fd2213b63c74017e7893c24d/bin" ${CONAN_BIN_DIRS})
set(CONAN_RES_DIRS "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/res"
			"/home/simon/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/res"
			"/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/res"
			"/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d/res" ${CONAN_RES_DIRS})
set(CONAN_FRAMEWORK_DIRS  ${CONAN_FRAMEWORK_DIRS})
set(CONAN_LIBS opencv_ml opencv_photo opencv_highgui opencv_videoio opencv_imgcodecs opencv_objdetect opencv_stitching opencv_dnn opencv_gapi opencv_video opencv_calib3d opencv_features2d opencv_flann opencv_imgproc opencv_core jasper IlmImfUtil-2_5 IlmImf-2_5 Imath-2_5 Half-2_5 IexMath-2_5 IlmThread-2_5 Iex-2_5 tiffxx tiff avdevice avfilter avformat avcodec swscale swresample postproc avutil quirc protoc protobuf ade jpeg deflate lzma jbig zstd webpdecoder webpdemux webpmux webp freetype openjp2 openh264 x264 x265 vpx mp3lame fdk-aac pulse-simple pulse pulsecommon-14.2 png16 bz2 iconv charset brotlidec-static brotlienc-static brotlicommon-static ssl crypto asound sndfile cap z vorbisenc vorbisfile vorbis opus FLAC++ FLAC ogg ${CONAN_LIBS})
set(CONAN_PKG_LIBS opencv_ml opencv_photo opencv_highgui opencv_videoio opencv_imgcodecs opencv_objdetect opencv_stitching opencv_dnn opencv_gapi opencv_video opencv_calib3d opencv_features2d opencv_flann opencv_imgproc opencv_core jasper IlmImfUtil-2_5 IlmImf-2_5 Imath-2_5 Half-2_5 IexMath-2_5 IlmThread-2_5 Iex-2_5 tiffxx tiff avdevice avfilter avformat avcodec swscale swresample postproc avutil quirc protoc protobuf ade jpeg deflate lzma jbig zstd webpdecoder webpdemux webpmux webp freetype openjp2 openh264 x264 x265 vpx mp3lame fdk-aac pulse-simple pulse pulsecommon-14.2 png16 bz2 iconv charset brotlidec-static brotlienc-static brotlicommon-static ssl crypto asound sndfile cap z vorbisenc vorbisfile vorbis opus FLAC++ FLAC ogg ${CONAN_PKG_LIBS})
set(CONAN_SYSTEM_LIBS gtk-x11-2.0 gdk-x11-2.0 pangocairo-1.0 atk-1.0 cairo gdk_pixbuf-2.0 gio-2.0 pangoft2-1.0 pango-1.0 gobject-2.0 glib-2.0 harfbuzz fontconfig freetype stdc++ va va-x11 va va-drm va vdpau X11 X11-xcb xcb fontenc ICE SM Xau Xaw7 Xt Xcomposite Xcursor Xdamage Xfixes Xdmcp Xext Xft Xi Xinerama xkbfile Xmu Xmuu Xpm Xrandr Xrender XRes Xss Xtst Xv XvMC Xxf86vm xcb-xkb xcb-icccm xcb-image xcb-shm xcb-keysyms xcb-randr xcb-render xcb-render-util xcb-shape xcb-sync xcb-xfixes xcb-xinerama xcb-util xcb-dri3 dl pthread rt m ${CONAN_SYSTEM_LIBS})
set(CONAN_FRAMEWORKS  ${CONAN_FRAMEWORKS})
set(CONAN_FRAMEWORKS_FOUND "")  # Will be filled later
set(CONAN_DEFINES "-DFLAC__NO_DLL"
			"-D_DEFAULT_SOURCE"
			"-D_BSD_SOURCE"
			"-DHAS_FCHOWN"
			"-DHAS_STICKY_DIR_BIT"
			"-D_REENTRANT"
			"-DLZMA_API_STATIC"
			"-DLIBJPEG_STATIC" ${CONAN_DEFINES})
set(CONAN_BUILD_MODULES_PATHS "/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/lib/cmake/protobuf/protobuf-generate.cmake"
			"/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/lib/cmake/protobuf/protobuf-module.cmake"
			"/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/lib/cmake/protobuf/protobuf-options.cmake" ${CONAN_BUILD_MODULES_PATHS})
set(CONAN_CMAKE_MODULE_PATH "/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/"
			"/home/simon/.conan/data/opencv/4.5.3/_/_/package/d08dde0a27a192529da0b441b86c4f16fe04503b/lib/cmake"
			"/home/simon/.conan/data/jasper/2.0.32/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/"
			"/home/simon/.conan/data/jasper/2.0.32/_/_/package/c0f14b099e9b359331abfa5be6f0d819983cea5b/lib/cmake"
			"/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/"
			"/home/simon/.conan/data/openexr/2.5.7/_/_/package/69698f6e2978013d4a5ee2a23b636a4c92eede84/lib/cmake"
			"/home/simon/.conan/data/libtiff/4.3.0/_/_/package/8302753393d88dd719b852c30dc285b2c793cd18/"
			"/home/simon/.conan/data/eigen/3.3.9/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/simon/.conan/data/ffmpeg/4.4/_/_/package/2770b46b8645e6555ada38043cbb4bf19de6ac0f/"
			"/home/simon/.conan/data/quirc/1.1/_/_/package/82c12ed3c6890f0bc8e954bd836c9da56b845711/"
			"/home/simon/.conan/data/gtk/system/_/_/package/cc5e5ca8b619d2dffaaeae1e4924ae3ce085a8f1/"
			"/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/"
			"/home/simon/.conan/data/protobuf/3.17.1/_/_/package/35f35211396802d98f502dcb5e3e87ae1b51c5f2/lib/cmake/protobuf"
			"/home/simon/.conan/data/ade/0.1.1f/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/"
			"/home/simon/.conan/data/ade/0.1.1f/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/lib/cmake"
			"/home/simon/.conan/data/libjpeg/9d/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/libdeflate/1.8/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/xz_utils/5.2.5/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib/cmake"
			"/home/simon/.conan/data/jbig/20160605/_/_/package/f2711434c4877d0e0aa8cdcc4da2a295d5d80304/"
			"/home/simon/.conan/data/zstd/1.5.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/libwebp/1.2.0/_/_/package/034afc24309e6fc60dbccb1dab5b0f9cad6ec656/"
			"/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/"
			"/home/simon/.conan/data/freetype/2.11.0/_/_/package/f1014dc4f9380132c471ceb778980949abf136d3/lib/cmake"
			"/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/openjpeg/2.4.0/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib/cmake"
			"/home/simon/.conan/data/openh264/2.1.1/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/"
			"/home/simon/.conan/data/libx264/20191217/_/_/package/52bfaf5780e9ee7fba3fa4a492156dbcf0dff057/"
			"/home/simon/.conan/data/libx265/3.4/_/_/package/75e903d40ade00f9adc18d07c1d36ecb3e595a39/"
			"/home/simon/.conan/data/libvpx/1.10.0/_/_/package/21d176c6d550e07ab7b17c1513a8ddb7fb960145/"
			"/home/simon/.conan/data/libmp3lame/3.100/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/libfdk_aac/2.0.2/_/_/package/b911f48570f9bb2902d9e83b2b9ebf9d376c8c56/"
			"/home/simon/.conan/data/pulseaudio/14.2/_/_/package/5fc8d29658aff7f1a5341125011d1c47807108ae/"
			"/home/simon/.conan/data/vaapi/system/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/simon/.conan/data/vdpau/system/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/simon/.conan/data/libpng/1.6.37/_/_/package/7929d8ecf29c60d74fd3c1f6cb78bbb3cb49c0c7/"
			"/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0/"
			"/home/simon/.conan/data/bzip2/1.0.8/_/_/package/da606cf731e334010b0bf6e85a2a6f891b9f36b0/lib/cmake"
			"/home/simon/.conan/data/libiconv/1.16/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/brotli/1.0.9/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/openssl/1.1.1l/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib/cmake"
			"/home/simon/.conan/data/libalsa/1.2.5.1/_/_/package/d48130e0dd76369b1338deb3b2372c5a649f9f2d/"
			"/home/simon/.conan/data/xorg/system/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/"
			"/home/simon/.conan/data/libsndfile/1.0.31/_/_/package/9384eaf0a0891af0df136cb9739770bd5d841e22/"
			"/home/simon/.conan/data/libcap/2.50/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/zlib/1.2.11/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/vorbis/1.3.7/_/_/package/033a9a186751ed382a6fa5fa2c844ad34b7c3302/"
			"/home/simon/.conan/data/opus/1.3.1/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/"
			"/home/simon/.conan/data/flac/1.3.3/_/_/package/b9bd28193f63cd74fd2213b63c74017e7893c24d/"
			"/home/simon/.conan/data/ogg/1.3.4/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/" ${CONAN_CMAKE_MODULE_PATH})

set(CONAN_CXX_FLAGS "-pthread ${CONAN_CXX_FLAGS}")
set(CONAN_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic,-znoexecstack -Wl,-Bsymbolic ${CONAN_SHARED_LINKER_FLAGS}")
set(CONAN_EXE_LINKER_FLAGS "-Wl,-Bsymbolic ${CONAN_EXE_LINKER_FLAGS}")
set(CONAN_C_FLAGS "-pthread ${CONAN_C_FLAGS}")

# Apple Frameworks
conan_find_apple_frameworks(CONAN_FRAMEWORKS_FOUND "${CONAN_FRAMEWORKS}" "" "")
# Append to aggregated values variable: Use CONAN_LIBS instead of CONAN_PKG_LIBS to include user appended vars
set(CONAN_LIBS ${CONAN_LIBS} ${CONAN_SYSTEM_LIBS} ${CONAN_FRAMEWORKS_FOUND})


###  Definition of macros and functions ###

macro(conan_define_targets)
    if(${CMAKE_VERSION} VERSION_LESS "3.1.2")
        message(FATAL_ERROR "TARGETS not supported by your CMake version!")
    endif()  # CMAKE > 3.x
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CONAN_CMD_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CONAN_CMD_C_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${CONAN_CMD_SHARED_LINKER_FLAGS}")


    set(_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES "${CONAN_SYSTEM_LIBS_OPENCV} ${CONAN_FRAMEWORKS_FOUND_OPENCV} CONAN_PKG::zlib CONAN_PKG::libjpeg CONAN_PKG::jasper CONAN_PKG::libpng CONAN_PKG::openexr CONAN_PKG::libtiff CONAN_PKG::eigen CONAN_PKG::ffmpeg CONAN_PKG::libwebp CONAN_PKG::quirc CONAN_PKG::gtk CONAN_PKG::protobuf CONAN_PKG::ade")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENCV_DEPENDENCIES "${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENCV}" "${CONAN_LIB_DIRS_OPENCV}"
                                  CONAN_PACKAGE_TARGETS_OPENCV "${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES}"
                                  "" opencv)
    set(_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_OPENCV_DEBUG} ${CONAN_FRAMEWORKS_FOUND_OPENCV_DEBUG} CONAN_PKG::zlib CONAN_PKG::libjpeg CONAN_PKG::jasper CONAN_PKG::libpng CONAN_PKG::openexr CONAN_PKG::libtiff CONAN_PKG::eigen CONAN_PKG::ffmpeg CONAN_PKG::libwebp CONAN_PKG::quirc CONAN_PKG::gtk CONAN_PKG::protobuf CONAN_PKG::ade")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENCV_DEBUG}" "${CONAN_LIB_DIRS_OPENCV_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_OPENCV_DEBUG "${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_DEBUG}"
                                  "debug" opencv)
    set(_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_OPENCV_RELEASE} ${CONAN_FRAMEWORKS_FOUND_OPENCV_RELEASE} CONAN_PKG::zlib CONAN_PKG::libjpeg CONAN_PKG::jasper CONAN_PKG::libpng CONAN_PKG::openexr CONAN_PKG::libtiff CONAN_PKG::eigen CONAN_PKG::ffmpeg CONAN_PKG::libwebp CONAN_PKG::quirc CONAN_PKG::gtk CONAN_PKG::protobuf CONAN_PKG::ade")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENCV_RELEASE}" "${CONAN_LIB_DIRS_OPENCV_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_OPENCV_RELEASE "${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_RELEASE}"
                                  "release" opencv)
    set(_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_OPENCV_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_OPENCV_RELWITHDEBINFO} CONAN_PKG::zlib CONAN_PKG::libjpeg CONAN_PKG::jasper CONAN_PKG::libpng CONAN_PKG::openexr CONAN_PKG::libtiff CONAN_PKG::eigen CONAN_PKG::ffmpeg CONAN_PKG::libwebp CONAN_PKG::quirc CONAN_PKG::gtk CONAN_PKG::protobuf CONAN_PKG::ade")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENCV_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_OPENCV_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_OPENCV_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" opencv)
    set(_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_OPENCV_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_OPENCV_MINSIZEREL} CONAN_PKG::zlib CONAN_PKG::libjpeg CONAN_PKG::jasper CONAN_PKG::libpng CONAN_PKG::openexr CONAN_PKG::libtiff CONAN_PKG::eigen CONAN_PKG::ffmpeg CONAN_PKG::libwebp CONAN_PKG::quirc CONAN_PKG::gtk CONAN_PKG::protobuf CONAN_PKG::ade")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENCV_MINSIZEREL}" "${CONAN_LIB_DIRS_OPENCV_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_OPENCV_MINSIZEREL "${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" opencv)

    add_library(CONAN_PKG::opencv INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::opencv PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_OPENCV} ${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENCV_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENCV_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENCV_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_OPENCV_RELEASE} ${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENCV_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENCV_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENCV_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_OPENCV_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENCV_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENCV_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENCV_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_OPENCV_MINSIZEREL} ${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENCV_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENCV_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENCV_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_OPENCV_DEBUG} ${_CONAN_PKG_LIBS_OPENCV_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENCV_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENCV_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENCV_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::opencv PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_OPENCV}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_OPENCV_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_OPENCV_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_OPENCV_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_OPENCV_DEBUG}>)
    set_property(TARGET CONAN_PKG::opencv PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_OPENCV}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_OPENCV_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_OPENCV_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_OPENCV_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_OPENCV_DEBUG}>)
    set_property(TARGET CONAN_PKG::opencv PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_OPENCV_LIST} ${CONAN_CXX_FLAGS_OPENCV_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_OPENCV_RELEASE_LIST} ${CONAN_CXX_FLAGS_OPENCV_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_OPENCV_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_OPENCV_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_OPENCV_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_OPENCV_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_OPENCV_DEBUG_LIST}  ${CONAN_CXX_FLAGS_OPENCV_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_JASPER_DEPENDENCIES "${CONAN_SYSTEM_LIBS_JASPER} ${CONAN_FRAMEWORKS_FOUND_JASPER} CONAN_PKG::libjpeg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_JASPER_DEPENDENCIES "${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_JASPER}" "${CONAN_LIB_DIRS_JASPER}"
                                  CONAN_PACKAGE_TARGETS_JASPER "${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES}"
                                  "" jasper)
    set(_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_JASPER_DEBUG} ${CONAN_FRAMEWORKS_FOUND_JASPER_DEBUG} CONAN_PKG::libjpeg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_JASPER_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_JASPER_DEBUG}" "${CONAN_LIB_DIRS_JASPER_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_JASPER_DEBUG "${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_DEBUG}"
                                  "debug" jasper)
    set(_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_JASPER_RELEASE} ${CONAN_FRAMEWORKS_FOUND_JASPER_RELEASE} CONAN_PKG::libjpeg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_JASPER_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_JASPER_RELEASE}" "${CONAN_LIB_DIRS_JASPER_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_JASPER_RELEASE "${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_RELEASE}"
                                  "release" jasper)
    set(_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_JASPER_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_JASPER_RELWITHDEBINFO} CONAN_PKG::libjpeg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_JASPER_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_JASPER_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_JASPER_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_JASPER_RELWITHDEBINFO "${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" jasper)
    set(_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_JASPER_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_JASPER_MINSIZEREL} CONAN_PKG::libjpeg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_JASPER_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_JASPER_MINSIZEREL}" "${CONAN_LIB_DIRS_JASPER_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_JASPER_MINSIZEREL "${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" jasper)

    add_library(CONAN_PKG::jasper INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::jasper PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_JASPER} ${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JASPER_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JASPER_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_JASPER_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_JASPER_RELEASE} ${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JASPER_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JASPER_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_JASPER_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_JASPER_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JASPER_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JASPER_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_JASPER_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_JASPER_MINSIZEREL} ${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JASPER_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JASPER_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_JASPER_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_JASPER_DEBUG} ${_CONAN_PKG_LIBS_JASPER_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JASPER_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JASPER_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_JASPER_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::jasper PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_JASPER}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_JASPER_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_JASPER_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_JASPER_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_JASPER_DEBUG}>)
    set_property(TARGET CONAN_PKG::jasper PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_JASPER}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_JASPER_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_JASPER_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_JASPER_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_JASPER_DEBUG}>)
    set_property(TARGET CONAN_PKG::jasper PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_JASPER_LIST} ${CONAN_CXX_FLAGS_JASPER_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_JASPER_RELEASE_LIST} ${CONAN_CXX_FLAGS_JASPER_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_JASPER_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_JASPER_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_JASPER_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_JASPER_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_JASPER_DEBUG_LIST}  ${CONAN_CXX_FLAGS_JASPER_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES "${CONAN_SYSTEM_LIBS_OPENEXR} ${CONAN_FRAMEWORKS_FOUND_OPENEXR} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES "${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENEXR}" "${CONAN_LIB_DIRS_OPENEXR}"
                                  CONAN_PACKAGE_TARGETS_OPENEXR "${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES}"
                                  "" openexr)
    set(_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_OPENEXR_DEBUG} ${CONAN_FRAMEWORKS_FOUND_OPENEXR_DEBUG} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENEXR_DEBUG}" "${CONAN_LIB_DIRS_OPENEXR_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_OPENEXR_DEBUG "${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_DEBUG}"
                                  "debug" openexr)
    set(_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_OPENEXR_RELEASE} ${CONAN_FRAMEWORKS_FOUND_OPENEXR_RELEASE} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENEXR_RELEASE}" "${CONAN_LIB_DIRS_OPENEXR_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_OPENEXR_RELEASE "${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_RELEASE}"
                                  "release" openexr)
    set(_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_OPENEXR_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_OPENEXR_RELWITHDEBINFO} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENEXR_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_OPENEXR_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_OPENEXR_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" openexr)
    set(_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_OPENEXR_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_OPENEXR_MINSIZEREL} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENEXR_MINSIZEREL}" "${CONAN_LIB_DIRS_OPENEXR_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_OPENEXR_MINSIZEREL "${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" openexr)

    add_library(CONAN_PKG::openexr INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::openexr PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_OPENEXR} ${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENEXR_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENEXR_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENEXR_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_OPENEXR_RELEASE} ${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENEXR_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENEXR_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENEXR_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_OPENEXR_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENEXR_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENEXR_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENEXR_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_OPENEXR_MINSIZEREL} ${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENEXR_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENEXR_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENEXR_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_OPENEXR_DEBUG} ${_CONAN_PKG_LIBS_OPENEXR_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENEXR_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENEXR_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENEXR_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::openexr PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_OPENEXR}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_OPENEXR_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_OPENEXR_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_OPENEXR_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_OPENEXR_DEBUG}>)
    set_property(TARGET CONAN_PKG::openexr PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_OPENEXR}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_OPENEXR_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_OPENEXR_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_OPENEXR_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_OPENEXR_DEBUG}>)
    set_property(TARGET CONAN_PKG::openexr PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_OPENEXR_LIST} ${CONAN_CXX_FLAGS_OPENEXR_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_OPENEXR_RELEASE_LIST} ${CONAN_CXX_FLAGS_OPENEXR_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_OPENEXR_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_OPENEXR_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_OPENEXR_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_OPENEXR_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_OPENEXR_DEBUG_LIST}  ${CONAN_CXX_FLAGS_OPENEXR_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBTIFF} ${CONAN_FRAMEWORKS_FOUND_LIBTIFF} CONAN_PKG::zlib CONAN_PKG::libdeflate CONAN_PKG::xz_utils CONAN_PKG::libjpeg CONAN_PKG::jbig CONAN_PKG::zstd CONAN_PKG::libwebp")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBTIFF}" "${CONAN_LIB_DIRS_LIBTIFF}"
                                  CONAN_PACKAGE_TARGETS_LIBTIFF "${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES}"
                                  "" libtiff)
    set(_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBTIFF_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBTIFF_DEBUG} CONAN_PKG::zlib CONAN_PKG::libdeflate CONAN_PKG::xz_utils CONAN_PKG::libjpeg CONAN_PKG::jbig CONAN_PKG::zstd CONAN_PKG::libwebp")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBTIFF_DEBUG}" "${CONAN_LIB_DIRS_LIBTIFF_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBTIFF_DEBUG "${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_DEBUG}"
                                  "debug" libtiff)
    set(_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBTIFF_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBTIFF_RELEASE} CONAN_PKG::zlib CONAN_PKG::libdeflate CONAN_PKG::xz_utils CONAN_PKG::libjpeg CONAN_PKG::jbig CONAN_PKG::zstd CONAN_PKG::libwebp")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBTIFF_RELEASE}" "${CONAN_LIB_DIRS_LIBTIFF_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBTIFF_RELEASE "${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_RELEASE}"
                                  "release" libtiff)
    set(_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBTIFF_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBTIFF_RELWITHDEBINFO} CONAN_PKG::zlib CONAN_PKG::libdeflate CONAN_PKG::xz_utils CONAN_PKG::libjpeg CONAN_PKG::jbig CONAN_PKG::zstd CONAN_PKG::libwebp")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBTIFF_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBTIFF_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBTIFF_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libtiff)
    set(_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBTIFF_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBTIFF_MINSIZEREL} CONAN_PKG::zlib CONAN_PKG::libdeflate CONAN_PKG::xz_utils CONAN_PKG::libjpeg CONAN_PKG::jbig CONAN_PKG::zstd CONAN_PKG::libwebp")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBTIFF_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBTIFF_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBTIFF_MINSIZEREL "${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libtiff)

    add_library(CONAN_PKG::libtiff INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libtiff PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBTIFF} ${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBTIFF_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBTIFF_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBTIFF_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBTIFF_RELEASE} ${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBTIFF_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBTIFF_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBTIFF_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBTIFF_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBTIFF_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBTIFF_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBTIFF_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBTIFF_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBTIFF_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBTIFF_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBTIFF_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBTIFF_DEBUG} ${_CONAN_PKG_LIBS_LIBTIFF_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBTIFF_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBTIFF_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBTIFF_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libtiff PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBTIFF}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBTIFF_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBTIFF_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBTIFF_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBTIFF_DEBUG}>)
    set_property(TARGET CONAN_PKG::libtiff PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBTIFF}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBTIFF_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBTIFF_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBTIFF_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBTIFF_DEBUG}>)
    set_property(TARGET CONAN_PKG::libtiff PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBTIFF_LIST} ${CONAN_CXX_FLAGS_LIBTIFF_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBTIFF_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBTIFF_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBTIFF_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBTIFF_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBTIFF_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBTIFF_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBTIFF_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBTIFF_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES "${CONAN_SYSTEM_LIBS_EIGEN} ${CONAN_FRAMEWORKS_FOUND_EIGEN} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_EIGEN_DEPENDENCIES "${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_EIGEN}" "${CONAN_LIB_DIRS_EIGEN}"
                                  CONAN_PACKAGE_TARGETS_EIGEN "${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES}"
                                  "" eigen)
    set(_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_EIGEN_DEBUG} ${CONAN_FRAMEWORKS_FOUND_EIGEN_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_EIGEN_DEBUG}" "${CONAN_LIB_DIRS_EIGEN_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_EIGEN_DEBUG "${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_DEBUG}"
                                  "debug" eigen)
    set(_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_EIGEN_RELEASE} ${CONAN_FRAMEWORKS_FOUND_EIGEN_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_EIGEN_RELEASE}" "${CONAN_LIB_DIRS_EIGEN_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_EIGEN_RELEASE "${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_RELEASE}"
                                  "release" eigen)
    set(_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_EIGEN_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_EIGEN_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_EIGEN_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_EIGEN_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_EIGEN_RELWITHDEBINFO "${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" eigen)
    set(_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_EIGEN_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_EIGEN_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_EIGEN_MINSIZEREL}" "${CONAN_LIB_DIRS_EIGEN_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_EIGEN_MINSIZEREL "${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" eigen)

    add_library(CONAN_PKG::eigen INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::eigen PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_EIGEN} ${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_EIGEN_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_EIGEN_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_EIGEN_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_EIGEN_RELEASE} ${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_EIGEN_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_EIGEN_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_EIGEN_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_EIGEN_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_EIGEN_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_EIGEN_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_EIGEN_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_EIGEN_MINSIZEREL} ${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_EIGEN_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_EIGEN_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_EIGEN_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_EIGEN_DEBUG} ${_CONAN_PKG_LIBS_EIGEN_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_EIGEN_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_EIGEN_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_EIGEN_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::eigen PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_EIGEN}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_EIGEN_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_EIGEN_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_EIGEN_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_EIGEN_DEBUG}>)
    set_property(TARGET CONAN_PKG::eigen PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_EIGEN}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_EIGEN_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_EIGEN_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_EIGEN_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_EIGEN_DEBUG}>)
    set_property(TARGET CONAN_PKG::eigen PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_EIGEN_LIST} ${CONAN_CXX_FLAGS_EIGEN_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_EIGEN_RELEASE_LIST} ${CONAN_CXX_FLAGS_EIGEN_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_EIGEN_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_EIGEN_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_EIGEN_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_EIGEN_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_EIGEN_DEBUG_LIST}  ${CONAN_CXX_FLAGS_EIGEN_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES "${CONAN_SYSTEM_LIBS_FFMPEG} ${CONAN_FRAMEWORKS_FOUND_FFMPEG} CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::xz_utils CONAN_PKG::libiconv CONAN_PKG::freetype CONAN_PKG::openjpeg CONAN_PKG::openh264 CONAN_PKG::vorbis CONAN_PKG::opus CONAN_PKG::libx264 CONAN_PKG::libx265 CONAN_PKG::libvpx CONAN_PKG::libmp3lame CONAN_PKG::libfdk_aac CONAN_PKG::libwebp CONAN_PKG::openssl CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::pulseaudio CONAN_PKG::vaapi CONAN_PKG::vdpau")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES "${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FFMPEG}" "${CONAN_LIB_DIRS_FFMPEG}"
                                  CONAN_PACKAGE_TARGETS_FFMPEG "${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES}"
                                  "" ffmpeg)
    set(_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_FFMPEG_DEBUG} ${CONAN_FRAMEWORKS_FOUND_FFMPEG_DEBUG} CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::xz_utils CONAN_PKG::libiconv CONAN_PKG::freetype CONAN_PKG::openjpeg CONAN_PKG::openh264 CONAN_PKG::vorbis CONAN_PKG::opus CONAN_PKG::libx264 CONAN_PKG::libx265 CONAN_PKG::libvpx CONAN_PKG::libmp3lame CONAN_PKG::libfdk_aac CONAN_PKG::libwebp CONAN_PKG::openssl CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::pulseaudio CONAN_PKG::vaapi CONAN_PKG::vdpau")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FFMPEG_DEBUG}" "${CONAN_LIB_DIRS_FFMPEG_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_FFMPEG_DEBUG "${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_DEBUG}"
                                  "debug" ffmpeg)
    set(_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_FFMPEG_RELEASE} ${CONAN_FRAMEWORKS_FOUND_FFMPEG_RELEASE} CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::xz_utils CONAN_PKG::libiconv CONAN_PKG::freetype CONAN_PKG::openjpeg CONAN_PKG::openh264 CONAN_PKG::vorbis CONAN_PKG::opus CONAN_PKG::libx264 CONAN_PKG::libx265 CONAN_PKG::libvpx CONAN_PKG::libmp3lame CONAN_PKG::libfdk_aac CONAN_PKG::libwebp CONAN_PKG::openssl CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::pulseaudio CONAN_PKG::vaapi CONAN_PKG::vdpau")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FFMPEG_RELEASE}" "${CONAN_LIB_DIRS_FFMPEG_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_FFMPEG_RELEASE "${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_RELEASE}"
                                  "release" ffmpeg)
    set(_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_FFMPEG_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_FFMPEG_RELWITHDEBINFO} CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::xz_utils CONAN_PKG::libiconv CONAN_PKG::freetype CONAN_PKG::openjpeg CONAN_PKG::openh264 CONAN_PKG::vorbis CONAN_PKG::opus CONAN_PKG::libx264 CONAN_PKG::libx265 CONAN_PKG::libvpx CONAN_PKG::libmp3lame CONAN_PKG::libfdk_aac CONAN_PKG::libwebp CONAN_PKG::openssl CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::pulseaudio CONAN_PKG::vaapi CONAN_PKG::vdpau")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FFMPEG_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_FFMPEG_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_FFMPEG_RELWITHDEBINFO "${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" ffmpeg)
    set(_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_FFMPEG_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_FFMPEG_MINSIZEREL} CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::xz_utils CONAN_PKG::libiconv CONAN_PKG::freetype CONAN_PKG::openjpeg CONAN_PKG::openh264 CONAN_PKG::vorbis CONAN_PKG::opus CONAN_PKG::libx264 CONAN_PKG::libx265 CONAN_PKG::libvpx CONAN_PKG::libmp3lame CONAN_PKG::libfdk_aac CONAN_PKG::libwebp CONAN_PKG::openssl CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::pulseaudio CONAN_PKG::vaapi CONAN_PKG::vdpau")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FFMPEG_MINSIZEREL}" "${CONAN_LIB_DIRS_FFMPEG_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_FFMPEG_MINSIZEREL "${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" ffmpeg)

    add_library(CONAN_PKG::ffmpeg INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::ffmpeg PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_FFMPEG} ${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FFMPEG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FFMPEG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FFMPEG_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_FFMPEG_RELEASE} ${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FFMPEG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FFMPEG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FFMPEG_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_FFMPEG_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FFMPEG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FFMPEG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FFMPEG_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_FFMPEG_MINSIZEREL} ${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FFMPEG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FFMPEG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FFMPEG_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_FFMPEG_DEBUG} ${_CONAN_PKG_LIBS_FFMPEG_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FFMPEG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FFMPEG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FFMPEG_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::ffmpeg PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_FFMPEG}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_FFMPEG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_FFMPEG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_FFMPEG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_FFMPEG_DEBUG}>)
    set_property(TARGET CONAN_PKG::ffmpeg PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_FFMPEG}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_FFMPEG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_FFMPEG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_FFMPEG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_FFMPEG_DEBUG}>)
    set_property(TARGET CONAN_PKG::ffmpeg PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_FFMPEG_LIST} ${CONAN_CXX_FLAGS_FFMPEG_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_FFMPEG_RELEASE_LIST} ${CONAN_CXX_FLAGS_FFMPEG_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_FFMPEG_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_FFMPEG_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_FFMPEG_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_FFMPEG_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_FFMPEG_DEBUG_LIST}  ${CONAN_CXX_FLAGS_FFMPEG_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES "${CONAN_SYSTEM_LIBS_QUIRC} ${CONAN_FRAMEWORKS_FOUND_QUIRC} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_QUIRC_DEPENDENCIES "${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_QUIRC}" "${CONAN_LIB_DIRS_QUIRC}"
                                  CONAN_PACKAGE_TARGETS_QUIRC "${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES}"
                                  "" quirc)
    set(_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_QUIRC_DEBUG} ${CONAN_FRAMEWORKS_FOUND_QUIRC_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_QUIRC_DEBUG}" "${CONAN_LIB_DIRS_QUIRC_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_QUIRC_DEBUG "${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_DEBUG}"
                                  "debug" quirc)
    set(_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_QUIRC_RELEASE} ${CONAN_FRAMEWORKS_FOUND_QUIRC_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_QUIRC_RELEASE}" "${CONAN_LIB_DIRS_QUIRC_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_QUIRC_RELEASE "${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_RELEASE}"
                                  "release" quirc)
    set(_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_QUIRC_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_QUIRC_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_QUIRC_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_QUIRC_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_QUIRC_RELWITHDEBINFO "${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" quirc)
    set(_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_QUIRC_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_QUIRC_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_QUIRC_MINSIZEREL}" "${CONAN_LIB_DIRS_QUIRC_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_QUIRC_MINSIZEREL "${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" quirc)

    add_library(CONAN_PKG::quirc INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::quirc PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_QUIRC} ${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_QUIRC_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_QUIRC_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_QUIRC_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_QUIRC_RELEASE} ${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_QUIRC_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_QUIRC_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_QUIRC_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_QUIRC_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_QUIRC_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_QUIRC_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_QUIRC_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_QUIRC_MINSIZEREL} ${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_QUIRC_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_QUIRC_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_QUIRC_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_QUIRC_DEBUG} ${_CONAN_PKG_LIBS_QUIRC_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_QUIRC_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_QUIRC_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_QUIRC_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::quirc PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_QUIRC}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_QUIRC_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_QUIRC_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_QUIRC_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_QUIRC_DEBUG}>)
    set_property(TARGET CONAN_PKG::quirc PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_QUIRC}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_QUIRC_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_QUIRC_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_QUIRC_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_QUIRC_DEBUG}>)
    set_property(TARGET CONAN_PKG::quirc PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_QUIRC_LIST} ${CONAN_CXX_FLAGS_QUIRC_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_QUIRC_RELEASE_LIST} ${CONAN_CXX_FLAGS_QUIRC_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_QUIRC_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_QUIRC_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_QUIRC_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_QUIRC_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_QUIRC_DEBUG_LIST}  ${CONAN_CXX_FLAGS_QUIRC_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_GTK_DEPENDENCIES "${CONAN_SYSTEM_LIBS_GTK} ${CONAN_FRAMEWORKS_FOUND_GTK} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_GTK_DEPENDENCIES "${_CONAN_PKG_LIBS_GTK_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_GTK}" "${CONAN_LIB_DIRS_GTK}"
                                  CONAN_PACKAGE_TARGETS_GTK "${_CONAN_PKG_LIBS_GTK_DEPENDENCIES}"
                                  "" gtk)
    set(_CONAN_PKG_LIBS_GTK_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_GTK_DEBUG} ${CONAN_FRAMEWORKS_FOUND_GTK_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_GTK_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_GTK_DEBUG}" "${CONAN_LIB_DIRS_GTK_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_GTK_DEBUG "${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_DEBUG}"
                                  "debug" gtk)
    set(_CONAN_PKG_LIBS_GTK_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_GTK_RELEASE} ${CONAN_FRAMEWORKS_FOUND_GTK_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_GTK_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_GTK_RELEASE}" "${CONAN_LIB_DIRS_GTK_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_GTK_RELEASE "${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_RELEASE}"
                                  "release" gtk)
    set(_CONAN_PKG_LIBS_GTK_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_GTK_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_GTK_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_GTK_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_GTK_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_GTK_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_GTK_RELWITHDEBINFO "${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" gtk)
    set(_CONAN_PKG_LIBS_GTK_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_GTK_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_GTK_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_GTK_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_GTK_MINSIZEREL}" "${CONAN_LIB_DIRS_GTK_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_GTK_MINSIZEREL "${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" gtk)

    add_library(CONAN_PKG::gtk INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::gtk PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_GTK} ${_CONAN_PKG_LIBS_GTK_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GTK_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GTK_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_GTK_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_GTK_RELEASE} ${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GTK_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GTK_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_GTK_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_GTK_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GTK_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GTK_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_GTK_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_GTK_MINSIZEREL} ${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GTK_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GTK_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_GTK_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_GTK_DEBUG} ${_CONAN_PKG_LIBS_GTK_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GTK_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_GTK_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_GTK_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::gtk PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_GTK}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_GTK_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_GTK_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_GTK_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_GTK_DEBUG}>)
    set_property(TARGET CONAN_PKG::gtk PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_GTK}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_GTK_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_GTK_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_GTK_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_GTK_DEBUG}>)
    set_property(TARGET CONAN_PKG::gtk PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_GTK_LIST} ${CONAN_CXX_FLAGS_GTK_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_GTK_RELEASE_LIST} ${CONAN_CXX_FLAGS_GTK_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_GTK_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_GTK_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_GTK_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_GTK_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_GTK_DEBUG_LIST}  ${CONAN_CXX_FLAGS_GTK_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES "${CONAN_SYSTEM_LIBS_PROTOBUF} ${CONAN_FRAMEWORKS_FOUND_PROTOBUF} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES "${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PROTOBUF}" "${CONAN_LIB_DIRS_PROTOBUF}"
                                  CONAN_PACKAGE_TARGETS_PROTOBUF "${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES}"
                                  "" protobuf)
    set(_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_PROTOBUF_DEBUG} ${CONAN_FRAMEWORKS_FOUND_PROTOBUF_DEBUG} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PROTOBUF_DEBUG}" "${CONAN_LIB_DIRS_PROTOBUF_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_PROTOBUF_DEBUG "${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_DEBUG}"
                                  "debug" protobuf)
    set(_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_PROTOBUF_RELEASE} ${CONAN_FRAMEWORKS_FOUND_PROTOBUF_RELEASE} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PROTOBUF_RELEASE}" "${CONAN_LIB_DIRS_PROTOBUF_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_PROTOBUF_RELEASE "${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_RELEASE}"
                                  "release" protobuf)
    set(_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_PROTOBUF_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_PROTOBUF_RELWITHDEBINFO} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PROTOBUF_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_PROTOBUF_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_PROTOBUF_RELWITHDEBINFO "${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" protobuf)
    set(_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_PROTOBUF_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_PROTOBUF_MINSIZEREL} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PROTOBUF_MINSIZEREL}" "${CONAN_LIB_DIRS_PROTOBUF_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_PROTOBUF_MINSIZEREL "${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" protobuf)

    add_library(CONAN_PKG::protobuf INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::protobuf PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_PROTOBUF} ${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PROTOBUF_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PROTOBUF_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PROTOBUF_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_PROTOBUF_RELEASE} ${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PROTOBUF_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PROTOBUF_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PROTOBUF_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_PROTOBUF_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PROTOBUF_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PROTOBUF_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PROTOBUF_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_PROTOBUF_MINSIZEREL} ${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PROTOBUF_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PROTOBUF_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PROTOBUF_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_PROTOBUF_DEBUG} ${_CONAN_PKG_LIBS_PROTOBUF_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PROTOBUF_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PROTOBUF_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PROTOBUF_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::protobuf PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_PROTOBUF}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_PROTOBUF_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_PROTOBUF_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_PROTOBUF_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_PROTOBUF_DEBUG}>)
    set_property(TARGET CONAN_PKG::protobuf PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_PROTOBUF}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_PROTOBUF_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_PROTOBUF_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_PROTOBUF_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_PROTOBUF_DEBUG}>)
    set_property(TARGET CONAN_PKG::protobuf PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_PROTOBUF_LIST} ${CONAN_CXX_FLAGS_PROTOBUF_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_PROTOBUF_RELEASE_LIST} ${CONAN_CXX_FLAGS_PROTOBUF_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_PROTOBUF_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_PROTOBUF_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_PROTOBUF_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_PROTOBUF_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_PROTOBUF_DEBUG_LIST}  ${CONAN_CXX_FLAGS_PROTOBUF_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_ADE_DEPENDENCIES "${CONAN_SYSTEM_LIBS_ADE} ${CONAN_FRAMEWORKS_FOUND_ADE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ADE_DEPENDENCIES "${_CONAN_PKG_LIBS_ADE_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ADE}" "${CONAN_LIB_DIRS_ADE}"
                                  CONAN_PACKAGE_TARGETS_ADE "${_CONAN_PKG_LIBS_ADE_DEPENDENCIES}"
                                  "" ade)
    set(_CONAN_PKG_LIBS_ADE_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_ADE_DEBUG} ${CONAN_FRAMEWORKS_FOUND_ADE_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ADE_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ADE_DEBUG}" "${CONAN_LIB_DIRS_ADE_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_ADE_DEBUG "${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_DEBUG}"
                                  "debug" ade)
    set(_CONAN_PKG_LIBS_ADE_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_ADE_RELEASE} ${CONAN_FRAMEWORKS_FOUND_ADE_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ADE_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ADE_RELEASE}" "${CONAN_LIB_DIRS_ADE_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_ADE_RELEASE "${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_RELEASE}"
                                  "release" ade)
    set(_CONAN_PKG_LIBS_ADE_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_ADE_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_ADE_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ADE_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ADE_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_ADE_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_ADE_RELWITHDEBINFO "${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" ade)
    set(_CONAN_PKG_LIBS_ADE_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_ADE_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_ADE_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ADE_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ADE_MINSIZEREL}" "${CONAN_LIB_DIRS_ADE_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_ADE_MINSIZEREL "${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" ade)

    add_library(CONAN_PKG::ade INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::ade PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_ADE} ${_CONAN_PKG_LIBS_ADE_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ADE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ADE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ADE_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_ADE_RELEASE} ${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ADE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ADE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ADE_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_ADE_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ADE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ADE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ADE_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_ADE_MINSIZEREL} ${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ADE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ADE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ADE_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_ADE_DEBUG} ${_CONAN_PKG_LIBS_ADE_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ADE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ADE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ADE_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::ade PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_ADE}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_ADE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_ADE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_ADE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_ADE_DEBUG}>)
    set_property(TARGET CONAN_PKG::ade PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_ADE}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_ADE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_ADE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_ADE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_ADE_DEBUG}>)
    set_property(TARGET CONAN_PKG::ade PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_ADE_LIST} ${CONAN_CXX_FLAGS_ADE_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_ADE_RELEASE_LIST} ${CONAN_CXX_FLAGS_ADE_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_ADE_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_ADE_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_ADE_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_ADE_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_ADE_DEBUG_LIST}  ${CONAN_CXX_FLAGS_ADE_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBJPEG} ${CONAN_FRAMEWORKS_FOUND_LIBJPEG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBJPEG}" "${CONAN_LIB_DIRS_LIBJPEG}"
                                  CONAN_PACKAGE_TARGETS_LIBJPEG "${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES}"
                                  "" libjpeg)
    set(_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBJPEG_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBJPEG_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBJPEG_DEBUG}" "${CONAN_LIB_DIRS_LIBJPEG_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBJPEG_DEBUG "${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_DEBUG}"
                                  "debug" libjpeg)
    set(_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBJPEG_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBJPEG_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBJPEG_RELEASE}" "${CONAN_LIB_DIRS_LIBJPEG_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBJPEG_RELEASE "${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_RELEASE}"
                                  "release" libjpeg)
    set(_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBJPEG_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBJPEG_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBJPEG_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBJPEG_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBJPEG_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libjpeg)
    set(_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBJPEG_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBJPEG_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBJPEG_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBJPEG_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBJPEG_MINSIZEREL "${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libjpeg)

    add_library(CONAN_PKG::libjpeg INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libjpeg PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBJPEG} ${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBJPEG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBJPEG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBJPEG_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBJPEG_RELEASE} ${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBJPEG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBJPEG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBJPEG_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBJPEG_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBJPEG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBJPEG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBJPEG_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBJPEG_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBJPEG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBJPEG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBJPEG_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBJPEG_DEBUG} ${_CONAN_PKG_LIBS_LIBJPEG_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBJPEG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBJPEG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBJPEG_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libjpeg PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBJPEG}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBJPEG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBJPEG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBJPEG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBJPEG_DEBUG}>)
    set_property(TARGET CONAN_PKG::libjpeg PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBJPEG}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBJPEG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBJPEG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBJPEG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBJPEG_DEBUG}>)
    set_property(TARGET CONAN_PKG::libjpeg PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBJPEG_LIST} ${CONAN_CXX_FLAGS_LIBJPEG_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBJPEG_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBJPEG_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBJPEG_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBJPEG_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBJPEG_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBJPEG_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBJPEG_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBJPEG_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBDEFLATE} ${CONAN_FRAMEWORKS_FOUND_LIBDEFLATE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBDEFLATE}" "${CONAN_LIB_DIRS_LIBDEFLATE}"
                                  CONAN_PACKAGE_TARGETS_LIBDEFLATE "${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES}"
                                  "" libdeflate)
    set(_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBDEFLATE_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBDEFLATE_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBDEFLATE_DEBUG}" "${CONAN_LIB_DIRS_LIBDEFLATE_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBDEFLATE_DEBUG "${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_DEBUG}"
                                  "debug" libdeflate)
    set(_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBDEFLATE_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBDEFLATE_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBDEFLATE_RELEASE}" "${CONAN_LIB_DIRS_LIBDEFLATE_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBDEFLATE_RELEASE "${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_RELEASE}"
                                  "release" libdeflate)
    set(_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBDEFLATE_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBDEFLATE_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBDEFLATE_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBDEFLATE_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBDEFLATE_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libdeflate)
    set(_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBDEFLATE_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBDEFLATE_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBDEFLATE_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBDEFLATE_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBDEFLATE_MINSIZEREL "${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libdeflate)

    add_library(CONAN_PKG::libdeflate INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libdeflate PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBDEFLATE} ${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBDEFLATE_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBDEFLATE_RELEASE} ${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBDEFLATE_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBDEFLATE_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBDEFLATE_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBDEFLATE_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBDEFLATE_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBDEFLATE_DEBUG} ${_CONAN_PKG_LIBS_LIBDEFLATE_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBDEFLATE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBDEFLATE_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libdeflate PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBDEFLATE}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBDEFLATE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBDEFLATE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBDEFLATE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBDEFLATE_DEBUG}>)
    set_property(TARGET CONAN_PKG::libdeflate PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBDEFLATE}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBDEFLATE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBDEFLATE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBDEFLATE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBDEFLATE_DEBUG}>)
    set_property(TARGET CONAN_PKG::libdeflate PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBDEFLATE_LIST} ${CONAN_CXX_FLAGS_LIBDEFLATE_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBDEFLATE_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBDEFLATE_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBDEFLATE_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBDEFLATE_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBDEFLATE_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBDEFLATE_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBDEFLATE_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBDEFLATE_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES "${CONAN_SYSTEM_LIBS_XZ_UTILS} ${CONAN_FRAMEWORKS_FOUND_XZ_UTILS} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES "${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_XZ_UTILS}" "${CONAN_LIB_DIRS_XZ_UTILS}"
                                  CONAN_PACKAGE_TARGETS_XZ_UTILS "${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES}"
                                  "" xz_utils)
    set(_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_XZ_UTILS_DEBUG} ${CONAN_FRAMEWORKS_FOUND_XZ_UTILS_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_XZ_UTILS_DEBUG}" "${CONAN_LIB_DIRS_XZ_UTILS_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_XZ_UTILS_DEBUG "${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_DEBUG}"
                                  "debug" xz_utils)
    set(_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_XZ_UTILS_RELEASE} ${CONAN_FRAMEWORKS_FOUND_XZ_UTILS_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_XZ_UTILS_RELEASE}" "${CONAN_LIB_DIRS_XZ_UTILS_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_XZ_UTILS_RELEASE "${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_RELEASE}"
                                  "release" xz_utils)
    set(_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_XZ_UTILS_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_XZ_UTILS_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_XZ_UTILS_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_XZ_UTILS_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_XZ_UTILS_RELWITHDEBINFO "${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" xz_utils)
    set(_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_XZ_UTILS_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_XZ_UTILS_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_XZ_UTILS_MINSIZEREL}" "${CONAN_LIB_DIRS_XZ_UTILS_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_XZ_UTILS_MINSIZEREL "${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" xz_utils)

    add_library(CONAN_PKG::xz_utils INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::xz_utils PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_XZ_UTILS} ${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_XZ_UTILS_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_XZ_UTILS_RELEASE} ${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_XZ_UTILS_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_XZ_UTILS_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_XZ_UTILS_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_XZ_UTILS_MINSIZEREL} ${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_XZ_UTILS_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_XZ_UTILS_DEBUG} ${_CONAN_PKG_LIBS_XZ_UTILS_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XZ_UTILS_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_XZ_UTILS_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::xz_utils PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_XZ_UTILS}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_XZ_UTILS_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_XZ_UTILS_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_XZ_UTILS_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_XZ_UTILS_DEBUG}>)
    set_property(TARGET CONAN_PKG::xz_utils PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_XZ_UTILS}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_XZ_UTILS_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_XZ_UTILS_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_XZ_UTILS_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_XZ_UTILS_DEBUG}>)
    set_property(TARGET CONAN_PKG::xz_utils PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_XZ_UTILS_LIST} ${CONAN_CXX_FLAGS_XZ_UTILS_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_XZ_UTILS_RELEASE_LIST} ${CONAN_CXX_FLAGS_XZ_UTILS_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_XZ_UTILS_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_XZ_UTILS_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_XZ_UTILS_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_XZ_UTILS_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_XZ_UTILS_DEBUG_LIST}  ${CONAN_CXX_FLAGS_XZ_UTILS_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_JBIG_DEPENDENCIES "${CONAN_SYSTEM_LIBS_JBIG} ${CONAN_FRAMEWORKS_FOUND_JBIG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_JBIG_DEPENDENCIES "${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_JBIG}" "${CONAN_LIB_DIRS_JBIG}"
                                  CONAN_PACKAGE_TARGETS_JBIG "${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES}"
                                  "" jbig)
    set(_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_JBIG_DEBUG} ${CONAN_FRAMEWORKS_FOUND_JBIG_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_JBIG_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_JBIG_DEBUG}" "${CONAN_LIB_DIRS_JBIG_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_JBIG_DEBUG "${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_DEBUG}"
                                  "debug" jbig)
    set(_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_JBIG_RELEASE} ${CONAN_FRAMEWORKS_FOUND_JBIG_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_JBIG_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_JBIG_RELEASE}" "${CONAN_LIB_DIRS_JBIG_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_JBIG_RELEASE "${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_RELEASE}"
                                  "release" jbig)
    set(_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_JBIG_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_JBIG_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_JBIG_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_JBIG_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_JBIG_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_JBIG_RELWITHDEBINFO "${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" jbig)
    set(_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_JBIG_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_JBIG_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_JBIG_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_JBIG_MINSIZEREL}" "${CONAN_LIB_DIRS_JBIG_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_JBIG_MINSIZEREL "${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" jbig)

    add_library(CONAN_PKG::jbig INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::jbig PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_JBIG} ${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JBIG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JBIG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_JBIG_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_JBIG_RELEASE} ${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JBIG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JBIG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_JBIG_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_JBIG_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JBIG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JBIG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_JBIG_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_JBIG_MINSIZEREL} ${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JBIG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JBIG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_JBIG_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_JBIG_DEBUG} ${_CONAN_PKG_LIBS_JBIG_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JBIG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_JBIG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_JBIG_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::jbig PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_JBIG}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_JBIG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_JBIG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_JBIG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_JBIG_DEBUG}>)
    set_property(TARGET CONAN_PKG::jbig PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_JBIG}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_JBIG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_JBIG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_JBIG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_JBIG_DEBUG}>)
    set_property(TARGET CONAN_PKG::jbig PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_JBIG_LIST} ${CONAN_CXX_FLAGS_JBIG_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_JBIG_RELEASE_LIST} ${CONAN_CXX_FLAGS_JBIG_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_JBIG_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_JBIG_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_JBIG_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_JBIG_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_JBIG_DEBUG_LIST}  ${CONAN_CXX_FLAGS_JBIG_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES "${CONAN_SYSTEM_LIBS_ZSTD} ${CONAN_FRAMEWORKS_FOUND_ZSTD} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZSTD_DEPENDENCIES "${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZSTD}" "${CONAN_LIB_DIRS_ZSTD}"
                                  CONAN_PACKAGE_TARGETS_ZSTD "${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES}"
                                  "" zstd)
    set(_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_ZSTD_DEBUG} ${CONAN_FRAMEWORKS_FOUND_ZSTD_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZSTD_DEBUG}" "${CONAN_LIB_DIRS_ZSTD_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_ZSTD_DEBUG "${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_DEBUG}"
                                  "debug" zstd)
    set(_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_ZSTD_RELEASE} ${CONAN_FRAMEWORKS_FOUND_ZSTD_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZSTD_RELEASE}" "${CONAN_LIB_DIRS_ZSTD_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_ZSTD_RELEASE "${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_RELEASE}"
                                  "release" zstd)
    set(_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_ZSTD_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_ZSTD_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZSTD_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_ZSTD_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_ZSTD_RELWITHDEBINFO "${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" zstd)
    set(_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_ZSTD_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_ZSTD_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZSTD_MINSIZEREL}" "${CONAN_LIB_DIRS_ZSTD_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_ZSTD_MINSIZEREL "${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" zstd)

    add_library(CONAN_PKG::zstd INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::zstd PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_ZSTD} ${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZSTD_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZSTD_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZSTD_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_ZSTD_RELEASE} ${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZSTD_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZSTD_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZSTD_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_ZSTD_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZSTD_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZSTD_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZSTD_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_ZSTD_MINSIZEREL} ${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZSTD_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZSTD_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZSTD_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_ZSTD_DEBUG} ${_CONAN_PKG_LIBS_ZSTD_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZSTD_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZSTD_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZSTD_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::zstd PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_ZSTD}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_ZSTD_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_ZSTD_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_ZSTD_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_ZSTD_DEBUG}>)
    set_property(TARGET CONAN_PKG::zstd PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_ZSTD}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_ZSTD_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_ZSTD_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_ZSTD_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_ZSTD_DEBUG}>)
    set_property(TARGET CONAN_PKG::zstd PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_ZSTD_LIST} ${CONAN_CXX_FLAGS_ZSTD_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_ZSTD_RELEASE_LIST} ${CONAN_CXX_FLAGS_ZSTD_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_ZSTD_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_ZSTD_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_ZSTD_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_ZSTD_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_ZSTD_DEBUG_LIST}  ${CONAN_CXX_FLAGS_ZSTD_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBWEBP} ${CONAN_FRAMEWORKS_FOUND_LIBWEBP} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBWEBP}" "${CONAN_LIB_DIRS_LIBWEBP}"
                                  CONAN_PACKAGE_TARGETS_LIBWEBP "${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES}"
                                  "" libwebp)
    set(_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBWEBP_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBWEBP_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBWEBP_DEBUG}" "${CONAN_LIB_DIRS_LIBWEBP_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBWEBP_DEBUG "${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_DEBUG}"
                                  "debug" libwebp)
    set(_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBWEBP_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBWEBP_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBWEBP_RELEASE}" "${CONAN_LIB_DIRS_LIBWEBP_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBWEBP_RELEASE "${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_RELEASE}"
                                  "release" libwebp)
    set(_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBWEBP_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBWEBP_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBWEBP_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBWEBP_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBWEBP_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libwebp)
    set(_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBWEBP_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBWEBP_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBWEBP_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBWEBP_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBWEBP_MINSIZEREL "${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libwebp)

    add_library(CONAN_PKG::libwebp INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libwebp PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBWEBP} ${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBWEBP_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBWEBP_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBWEBP_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBWEBP_RELEASE} ${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBWEBP_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBWEBP_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBWEBP_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBWEBP_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBWEBP_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBWEBP_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBWEBP_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBWEBP_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBWEBP_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBWEBP_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBWEBP_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBWEBP_DEBUG} ${_CONAN_PKG_LIBS_LIBWEBP_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBWEBP_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBWEBP_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBWEBP_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libwebp PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBWEBP}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBWEBP_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBWEBP_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBWEBP_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBWEBP_DEBUG}>)
    set_property(TARGET CONAN_PKG::libwebp PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBWEBP}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBWEBP_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBWEBP_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBWEBP_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBWEBP_DEBUG}>)
    set_property(TARGET CONAN_PKG::libwebp PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBWEBP_LIST} ${CONAN_CXX_FLAGS_LIBWEBP_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBWEBP_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBWEBP_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBWEBP_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBWEBP_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBWEBP_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBWEBP_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBWEBP_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBWEBP_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES "${CONAN_SYSTEM_LIBS_FREETYPE} ${CONAN_FRAMEWORKS_FOUND_FREETYPE} CONAN_PKG::libpng CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::brotli")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES "${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FREETYPE}" "${CONAN_LIB_DIRS_FREETYPE}"
                                  CONAN_PACKAGE_TARGETS_FREETYPE "${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES}"
                                  "" freetype)
    set(_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_FREETYPE_DEBUG} ${CONAN_FRAMEWORKS_FOUND_FREETYPE_DEBUG} CONAN_PKG::libpng CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::brotli")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FREETYPE_DEBUG}" "${CONAN_LIB_DIRS_FREETYPE_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_FREETYPE_DEBUG "${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_DEBUG}"
                                  "debug" freetype)
    set(_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_FREETYPE_RELEASE} ${CONAN_FRAMEWORKS_FOUND_FREETYPE_RELEASE} CONAN_PKG::libpng CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::brotli")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FREETYPE_RELEASE}" "${CONAN_LIB_DIRS_FREETYPE_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_FREETYPE_RELEASE "${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_RELEASE}"
                                  "release" freetype)
    set(_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_FREETYPE_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_FREETYPE_RELWITHDEBINFO} CONAN_PKG::libpng CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::brotli")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FREETYPE_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_FREETYPE_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_FREETYPE_RELWITHDEBINFO "${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" freetype)
    set(_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_FREETYPE_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_FREETYPE_MINSIZEREL} CONAN_PKG::libpng CONAN_PKG::zlib CONAN_PKG::bzip2 CONAN_PKG::brotli")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FREETYPE_MINSIZEREL}" "${CONAN_LIB_DIRS_FREETYPE_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_FREETYPE_MINSIZEREL "${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" freetype)

    add_library(CONAN_PKG::freetype INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::freetype PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_FREETYPE} ${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FREETYPE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FREETYPE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FREETYPE_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_FREETYPE_RELEASE} ${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FREETYPE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FREETYPE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FREETYPE_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_FREETYPE_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FREETYPE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FREETYPE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FREETYPE_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_FREETYPE_MINSIZEREL} ${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FREETYPE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FREETYPE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FREETYPE_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_FREETYPE_DEBUG} ${_CONAN_PKG_LIBS_FREETYPE_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FREETYPE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FREETYPE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FREETYPE_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::freetype PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_FREETYPE}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_FREETYPE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_FREETYPE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_FREETYPE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_FREETYPE_DEBUG}>)
    set_property(TARGET CONAN_PKG::freetype PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_FREETYPE}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_FREETYPE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_FREETYPE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_FREETYPE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_FREETYPE_DEBUG}>)
    set_property(TARGET CONAN_PKG::freetype PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_FREETYPE_LIST} ${CONAN_CXX_FLAGS_FREETYPE_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_FREETYPE_RELEASE_LIST} ${CONAN_CXX_FLAGS_FREETYPE_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_FREETYPE_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_FREETYPE_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_FREETYPE_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_FREETYPE_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_FREETYPE_DEBUG_LIST}  ${CONAN_CXX_FLAGS_FREETYPE_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES "${CONAN_SYSTEM_LIBS_OPENJPEG} ${CONAN_FRAMEWORKS_FOUND_OPENJPEG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES "${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENJPEG}" "${CONAN_LIB_DIRS_OPENJPEG}"
                                  CONAN_PACKAGE_TARGETS_OPENJPEG "${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES}"
                                  "" openjpeg)
    set(_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_OPENJPEG_DEBUG} ${CONAN_FRAMEWORKS_FOUND_OPENJPEG_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENJPEG_DEBUG}" "${CONAN_LIB_DIRS_OPENJPEG_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_OPENJPEG_DEBUG "${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_DEBUG}"
                                  "debug" openjpeg)
    set(_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_OPENJPEG_RELEASE} ${CONAN_FRAMEWORKS_FOUND_OPENJPEG_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENJPEG_RELEASE}" "${CONAN_LIB_DIRS_OPENJPEG_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_OPENJPEG_RELEASE "${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_RELEASE}"
                                  "release" openjpeg)
    set(_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_OPENJPEG_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_OPENJPEG_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENJPEG_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_OPENJPEG_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_OPENJPEG_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" openjpeg)
    set(_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_OPENJPEG_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_OPENJPEG_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENJPEG_MINSIZEREL}" "${CONAN_LIB_DIRS_OPENJPEG_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_OPENJPEG_MINSIZEREL "${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" openjpeg)

    add_library(CONAN_PKG::openjpeg INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::openjpeg PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_OPENJPEG} ${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENJPEG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENJPEG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENJPEG_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_OPENJPEG_RELEASE} ${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENJPEG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENJPEG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENJPEG_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_OPENJPEG_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENJPEG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENJPEG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENJPEG_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_OPENJPEG_MINSIZEREL} ${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENJPEG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENJPEG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENJPEG_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_OPENJPEG_DEBUG} ${_CONAN_PKG_LIBS_OPENJPEG_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENJPEG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENJPEG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENJPEG_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::openjpeg PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_OPENJPEG}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_OPENJPEG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_OPENJPEG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_OPENJPEG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_OPENJPEG_DEBUG}>)
    set_property(TARGET CONAN_PKG::openjpeg PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_OPENJPEG}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_OPENJPEG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_OPENJPEG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_OPENJPEG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_OPENJPEG_DEBUG}>)
    set_property(TARGET CONAN_PKG::openjpeg PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_OPENJPEG_LIST} ${CONAN_CXX_FLAGS_OPENJPEG_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_OPENJPEG_RELEASE_LIST} ${CONAN_CXX_FLAGS_OPENJPEG_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_OPENJPEG_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_OPENJPEG_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_OPENJPEG_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_OPENJPEG_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_OPENJPEG_DEBUG_LIST}  ${CONAN_CXX_FLAGS_OPENJPEG_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES "${CONAN_SYSTEM_LIBS_OPENH264} ${CONAN_FRAMEWORKS_FOUND_OPENH264} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENH264_DEPENDENCIES "${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENH264}" "${CONAN_LIB_DIRS_OPENH264}"
                                  CONAN_PACKAGE_TARGETS_OPENH264 "${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES}"
                                  "" openh264)
    set(_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_OPENH264_DEBUG} ${CONAN_FRAMEWORKS_FOUND_OPENH264_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENH264_DEBUG}" "${CONAN_LIB_DIRS_OPENH264_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_OPENH264_DEBUG "${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_DEBUG}"
                                  "debug" openh264)
    set(_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_OPENH264_RELEASE} ${CONAN_FRAMEWORKS_FOUND_OPENH264_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENH264_RELEASE}" "${CONAN_LIB_DIRS_OPENH264_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_OPENH264_RELEASE "${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_RELEASE}"
                                  "release" openh264)
    set(_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_OPENH264_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_OPENH264_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENH264_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_OPENH264_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_OPENH264_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" openh264)
    set(_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_OPENH264_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_OPENH264_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENH264_MINSIZEREL}" "${CONAN_LIB_DIRS_OPENH264_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_OPENH264_MINSIZEREL "${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" openh264)

    add_library(CONAN_PKG::openh264 INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::openh264 PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_OPENH264} ${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENH264_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENH264_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENH264_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_OPENH264_RELEASE} ${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENH264_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENH264_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENH264_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_OPENH264_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENH264_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENH264_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENH264_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_OPENH264_MINSIZEREL} ${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENH264_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENH264_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENH264_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_OPENH264_DEBUG} ${_CONAN_PKG_LIBS_OPENH264_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENH264_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENH264_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENH264_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::openh264 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_OPENH264}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_OPENH264_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_OPENH264_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_OPENH264_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_OPENH264_DEBUG}>)
    set_property(TARGET CONAN_PKG::openh264 PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_OPENH264}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_OPENH264_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_OPENH264_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_OPENH264_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_OPENH264_DEBUG}>)
    set_property(TARGET CONAN_PKG::openh264 PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_OPENH264_LIST} ${CONAN_CXX_FLAGS_OPENH264_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_OPENH264_RELEASE_LIST} ${CONAN_CXX_FLAGS_OPENH264_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_OPENH264_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_OPENH264_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_OPENH264_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_OPENH264_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_OPENH264_DEBUG_LIST}  ${CONAN_CXX_FLAGS_OPENH264_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBX264} ${CONAN_FRAMEWORKS_FOUND_LIBX264} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBX264_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBX264}" "${CONAN_LIB_DIRS_LIBX264}"
                                  CONAN_PACKAGE_TARGETS_LIBX264 "${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES}"
                                  "" libx264)
    set(_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBX264_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBX264_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBX264_DEBUG}" "${CONAN_LIB_DIRS_LIBX264_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBX264_DEBUG "${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_DEBUG}"
                                  "debug" libx264)
    set(_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBX264_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBX264_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBX264_RELEASE}" "${CONAN_LIB_DIRS_LIBX264_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBX264_RELEASE "${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_RELEASE}"
                                  "release" libx264)
    set(_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBX264_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBX264_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBX264_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBX264_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBX264_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libx264)
    set(_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBX264_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBX264_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBX264_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBX264_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBX264_MINSIZEREL "${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libx264)

    add_library(CONAN_PKG::libx264 INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libx264 PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBX264} ${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX264_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX264_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBX264_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBX264_RELEASE} ${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX264_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX264_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBX264_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBX264_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX264_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX264_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBX264_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBX264_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX264_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX264_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBX264_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBX264_DEBUG} ${_CONAN_PKG_LIBS_LIBX264_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX264_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX264_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBX264_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libx264 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBX264}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBX264_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBX264_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBX264_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBX264_DEBUG}>)
    set_property(TARGET CONAN_PKG::libx264 PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBX264}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBX264_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBX264_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBX264_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBX264_DEBUG}>)
    set_property(TARGET CONAN_PKG::libx264 PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBX264_LIST} ${CONAN_CXX_FLAGS_LIBX264_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBX264_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBX264_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBX264_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBX264_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBX264_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBX264_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBX264_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBX264_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBX265} ${CONAN_FRAMEWORKS_FOUND_LIBX265} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBX265_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBX265}" "${CONAN_LIB_DIRS_LIBX265}"
                                  CONAN_PACKAGE_TARGETS_LIBX265 "${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES}"
                                  "" libx265)
    set(_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBX265_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBX265_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBX265_DEBUG}" "${CONAN_LIB_DIRS_LIBX265_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBX265_DEBUG "${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_DEBUG}"
                                  "debug" libx265)
    set(_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBX265_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBX265_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBX265_RELEASE}" "${CONAN_LIB_DIRS_LIBX265_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBX265_RELEASE "${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_RELEASE}"
                                  "release" libx265)
    set(_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBX265_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBX265_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBX265_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBX265_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBX265_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libx265)
    set(_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBX265_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBX265_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBX265_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBX265_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBX265_MINSIZEREL "${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libx265)

    add_library(CONAN_PKG::libx265 INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libx265 PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBX265} ${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX265_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX265_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBX265_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBX265_RELEASE} ${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX265_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX265_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBX265_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBX265_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX265_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX265_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBX265_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBX265_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX265_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX265_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBX265_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBX265_DEBUG} ${_CONAN_PKG_LIBS_LIBX265_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX265_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBX265_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBX265_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libx265 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBX265}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBX265_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBX265_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBX265_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBX265_DEBUG}>)
    set_property(TARGET CONAN_PKG::libx265 PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBX265}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBX265_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBX265_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBX265_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBX265_DEBUG}>)
    set_property(TARGET CONAN_PKG::libx265 PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBX265_LIST} ${CONAN_CXX_FLAGS_LIBX265_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBX265_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBX265_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBX265_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBX265_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBX265_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBX265_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBX265_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBX265_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBVPX} ${CONAN_FRAMEWORKS_FOUND_LIBVPX} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBVPX}" "${CONAN_LIB_DIRS_LIBVPX}"
                                  CONAN_PACKAGE_TARGETS_LIBVPX "${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES}"
                                  "" libvpx)
    set(_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBVPX_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBVPX_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBVPX_DEBUG}" "${CONAN_LIB_DIRS_LIBVPX_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBVPX_DEBUG "${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_DEBUG}"
                                  "debug" libvpx)
    set(_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBVPX_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBVPX_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBVPX_RELEASE}" "${CONAN_LIB_DIRS_LIBVPX_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBVPX_RELEASE "${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_RELEASE}"
                                  "release" libvpx)
    set(_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBVPX_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBVPX_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBVPX_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBVPX_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBVPX_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libvpx)
    set(_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBVPX_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBVPX_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBVPX_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBVPX_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBVPX_MINSIZEREL "${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libvpx)

    add_library(CONAN_PKG::libvpx INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libvpx PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBVPX} ${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBVPX_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBVPX_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBVPX_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBVPX_RELEASE} ${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBVPX_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBVPX_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBVPX_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBVPX_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBVPX_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBVPX_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBVPX_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBVPX_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBVPX_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBVPX_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBVPX_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBVPX_DEBUG} ${_CONAN_PKG_LIBS_LIBVPX_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBVPX_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBVPX_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBVPX_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libvpx PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBVPX}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBVPX_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBVPX_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBVPX_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBVPX_DEBUG}>)
    set_property(TARGET CONAN_PKG::libvpx PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBVPX}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBVPX_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBVPX_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBVPX_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBVPX_DEBUG}>)
    set_property(TARGET CONAN_PKG::libvpx PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBVPX_LIST} ${CONAN_CXX_FLAGS_LIBVPX_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBVPX_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBVPX_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBVPX_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBVPX_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBVPX_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBVPX_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBVPX_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBVPX_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBMP3LAME} ${CONAN_FRAMEWORKS_FOUND_LIBMP3LAME} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBMP3LAME}" "${CONAN_LIB_DIRS_LIBMP3LAME}"
                                  CONAN_PACKAGE_TARGETS_LIBMP3LAME "${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES}"
                                  "" libmp3lame)
    set(_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBMP3LAME_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBMP3LAME_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBMP3LAME_DEBUG}" "${CONAN_LIB_DIRS_LIBMP3LAME_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBMP3LAME_DEBUG "${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_DEBUG}"
                                  "debug" libmp3lame)
    set(_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBMP3LAME_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBMP3LAME_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBMP3LAME_RELEASE}" "${CONAN_LIB_DIRS_LIBMP3LAME_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBMP3LAME_RELEASE "${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_RELEASE}"
                                  "release" libmp3lame)
    set(_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBMP3LAME_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBMP3LAME_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBMP3LAME_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBMP3LAME_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBMP3LAME_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libmp3lame)
    set(_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBMP3LAME_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBMP3LAME_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBMP3LAME_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBMP3LAME_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBMP3LAME_MINSIZEREL "${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libmp3lame)

    add_library(CONAN_PKG::libmp3lame INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libmp3lame PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBMP3LAME} ${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBMP3LAME_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBMP3LAME_RELEASE} ${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBMP3LAME_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBMP3LAME_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBMP3LAME_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBMP3LAME_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBMP3LAME_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBMP3LAME_DEBUG} ${_CONAN_PKG_LIBS_LIBMP3LAME_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBMP3LAME_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBMP3LAME_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libmp3lame PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBMP3LAME}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBMP3LAME_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBMP3LAME_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBMP3LAME_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBMP3LAME_DEBUG}>)
    set_property(TARGET CONAN_PKG::libmp3lame PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBMP3LAME}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBMP3LAME_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBMP3LAME_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBMP3LAME_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBMP3LAME_DEBUG}>)
    set_property(TARGET CONAN_PKG::libmp3lame PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBMP3LAME_LIST} ${CONAN_CXX_FLAGS_LIBMP3LAME_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBMP3LAME_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBMP3LAME_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBMP3LAME_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBMP3LAME_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBMP3LAME_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBMP3LAME_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBMP3LAME_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBMP3LAME_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBFDK_AAC} ${CONAN_FRAMEWORKS_FOUND_LIBFDK_AAC} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBFDK_AAC}" "${CONAN_LIB_DIRS_LIBFDK_AAC}"
                                  CONAN_PACKAGE_TARGETS_LIBFDK_AAC "${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES}"
                                  "" libfdk_aac)
    set(_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBFDK_AAC_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBFDK_AAC_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBFDK_AAC_DEBUG}" "${CONAN_LIB_DIRS_LIBFDK_AAC_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBFDK_AAC_DEBUG "${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_DEBUG}"
                                  "debug" libfdk_aac)
    set(_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBFDK_AAC_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBFDK_AAC_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBFDK_AAC_RELEASE}" "${CONAN_LIB_DIRS_LIBFDK_AAC_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBFDK_AAC_RELEASE "${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_RELEASE}"
                                  "release" libfdk_aac)
    set(_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBFDK_AAC_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBFDK_AAC_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBFDK_AAC_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBFDK_AAC_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBFDK_AAC_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libfdk_aac)
    set(_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBFDK_AAC_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBFDK_AAC_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBFDK_AAC_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBFDK_AAC_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBFDK_AAC_MINSIZEREL "${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libfdk_aac)

    add_library(CONAN_PKG::libfdk_aac INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libfdk_aac PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBFDK_AAC} ${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBFDK_AAC_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBFDK_AAC_RELEASE} ${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBFDK_AAC_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBFDK_AAC_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBFDK_AAC_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBFDK_AAC_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBFDK_AAC_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBFDK_AAC_DEBUG} ${_CONAN_PKG_LIBS_LIBFDK_AAC_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBFDK_AAC_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBFDK_AAC_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libfdk_aac PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBFDK_AAC}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBFDK_AAC_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBFDK_AAC_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBFDK_AAC_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBFDK_AAC_DEBUG}>)
    set_property(TARGET CONAN_PKG::libfdk_aac PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBFDK_AAC}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBFDK_AAC_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBFDK_AAC_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBFDK_AAC_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBFDK_AAC_DEBUG}>)
    set_property(TARGET CONAN_PKG::libfdk_aac PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBFDK_AAC_LIST} ${CONAN_CXX_FLAGS_LIBFDK_AAC_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBFDK_AAC_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBFDK_AAC_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBFDK_AAC_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBFDK_AAC_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBFDK_AAC_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBFDK_AAC_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBFDK_AAC_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBFDK_AAC_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES "${CONAN_SYSTEM_LIBS_PULSEAUDIO} ${CONAN_FRAMEWORKS_FOUND_PULSEAUDIO} CONAN_PKG::libiconv CONAN_PKG::libsndfile CONAN_PKG::libcap CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::openssl")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES "${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PULSEAUDIO}" "${CONAN_LIB_DIRS_PULSEAUDIO}"
                                  CONAN_PACKAGE_TARGETS_PULSEAUDIO "${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES}"
                                  "" pulseaudio)
    set(_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_PULSEAUDIO_DEBUG} ${CONAN_FRAMEWORKS_FOUND_PULSEAUDIO_DEBUG} CONAN_PKG::libiconv CONAN_PKG::libsndfile CONAN_PKG::libcap CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::openssl")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PULSEAUDIO_DEBUG}" "${CONAN_LIB_DIRS_PULSEAUDIO_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_PULSEAUDIO_DEBUG "${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_DEBUG}"
                                  "debug" pulseaudio)
    set(_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_PULSEAUDIO_RELEASE} ${CONAN_FRAMEWORKS_FOUND_PULSEAUDIO_RELEASE} CONAN_PKG::libiconv CONAN_PKG::libsndfile CONAN_PKG::libcap CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::openssl")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PULSEAUDIO_RELEASE}" "${CONAN_LIB_DIRS_PULSEAUDIO_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_PULSEAUDIO_RELEASE "${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_RELEASE}"
                                  "release" pulseaudio)
    set(_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_PULSEAUDIO_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_PULSEAUDIO_RELWITHDEBINFO} CONAN_PKG::libiconv CONAN_PKG::libsndfile CONAN_PKG::libcap CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::openssl")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PULSEAUDIO_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_PULSEAUDIO_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_PULSEAUDIO_RELWITHDEBINFO "${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" pulseaudio)
    set(_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_PULSEAUDIO_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_PULSEAUDIO_MINSIZEREL} CONAN_PKG::libiconv CONAN_PKG::libsndfile CONAN_PKG::libcap CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::openssl")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_PULSEAUDIO_MINSIZEREL}" "${CONAN_LIB_DIRS_PULSEAUDIO_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_PULSEAUDIO_MINSIZEREL "${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" pulseaudio)

    add_library(CONAN_PKG::pulseaudio INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::pulseaudio PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_PULSEAUDIO} ${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PULSEAUDIO_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_PULSEAUDIO_RELEASE} ${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PULSEAUDIO_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_PULSEAUDIO_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PULSEAUDIO_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_PULSEAUDIO_MINSIZEREL} ${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PULSEAUDIO_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_PULSEAUDIO_DEBUG} ${_CONAN_PKG_LIBS_PULSEAUDIO_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_PULSEAUDIO_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_PULSEAUDIO_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::pulseaudio PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_PULSEAUDIO}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_PULSEAUDIO_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_PULSEAUDIO_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_PULSEAUDIO_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_PULSEAUDIO_DEBUG}>)
    set_property(TARGET CONAN_PKG::pulseaudio PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_PULSEAUDIO}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_PULSEAUDIO_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_PULSEAUDIO_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_PULSEAUDIO_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_PULSEAUDIO_DEBUG}>)
    set_property(TARGET CONAN_PKG::pulseaudio PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_PULSEAUDIO_LIST} ${CONAN_CXX_FLAGS_PULSEAUDIO_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_PULSEAUDIO_RELEASE_LIST} ${CONAN_CXX_FLAGS_PULSEAUDIO_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_PULSEAUDIO_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_PULSEAUDIO_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_PULSEAUDIO_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_PULSEAUDIO_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_PULSEAUDIO_DEBUG_LIST}  ${CONAN_CXX_FLAGS_PULSEAUDIO_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES "${CONAN_SYSTEM_LIBS_VAAPI} ${CONAN_FRAMEWORKS_FOUND_VAAPI} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VAAPI_DEPENDENCIES "${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VAAPI}" "${CONAN_LIB_DIRS_VAAPI}"
                                  CONAN_PACKAGE_TARGETS_VAAPI "${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES}"
                                  "" vaapi)
    set(_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_VAAPI_DEBUG} ${CONAN_FRAMEWORKS_FOUND_VAAPI_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VAAPI_DEBUG}" "${CONAN_LIB_DIRS_VAAPI_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_VAAPI_DEBUG "${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_DEBUG}"
                                  "debug" vaapi)
    set(_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_VAAPI_RELEASE} ${CONAN_FRAMEWORKS_FOUND_VAAPI_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VAAPI_RELEASE}" "${CONAN_LIB_DIRS_VAAPI_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_VAAPI_RELEASE "${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_RELEASE}"
                                  "release" vaapi)
    set(_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_VAAPI_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_VAAPI_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VAAPI_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_VAAPI_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_VAAPI_RELWITHDEBINFO "${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" vaapi)
    set(_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_VAAPI_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_VAAPI_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VAAPI_MINSIZEREL}" "${CONAN_LIB_DIRS_VAAPI_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_VAAPI_MINSIZEREL "${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" vaapi)

    add_library(CONAN_PKG::vaapi INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::vaapi PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_VAAPI} ${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VAAPI_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VAAPI_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VAAPI_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_VAAPI_RELEASE} ${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VAAPI_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VAAPI_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VAAPI_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_VAAPI_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VAAPI_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VAAPI_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VAAPI_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_VAAPI_MINSIZEREL} ${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VAAPI_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VAAPI_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VAAPI_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_VAAPI_DEBUG} ${_CONAN_PKG_LIBS_VAAPI_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VAAPI_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VAAPI_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VAAPI_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::vaapi PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_VAAPI}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_VAAPI_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_VAAPI_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_VAAPI_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_VAAPI_DEBUG}>)
    set_property(TARGET CONAN_PKG::vaapi PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_VAAPI}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_VAAPI_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_VAAPI_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_VAAPI_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_VAAPI_DEBUG}>)
    set_property(TARGET CONAN_PKG::vaapi PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_VAAPI_LIST} ${CONAN_CXX_FLAGS_VAAPI_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_VAAPI_RELEASE_LIST} ${CONAN_CXX_FLAGS_VAAPI_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_VAAPI_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_VAAPI_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_VAAPI_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_VAAPI_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_VAAPI_DEBUG_LIST}  ${CONAN_CXX_FLAGS_VAAPI_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES "${CONAN_SYSTEM_LIBS_VDPAU} ${CONAN_FRAMEWORKS_FOUND_VDPAU} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VDPAU_DEPENDENCIES "${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VDPAU}" "${CONAN_LIB_DIRS_VDPAU}"
                                  CONAN_PACKAGE_TARGETS_VDPAU "${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES}"
                                  "" vdpau)
    set(_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_VDPAU_DEBUG} ${CONAN_FRAMEWORKS_FOUND_VDPAU_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VDPAU_DEBUG}" "${CONAN_LIB_DIRS_VDPAU_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_VDPAU_DEBUG "${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_DEBUG}"
                                  "debug" vdpau)
    set(_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_VDPAU_RELEASE} ${CONAN_FRAMEWORKS_FOUND_VDPAU_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VDPAU_RELEASE}" "${CONAN_LIB_DIRS_VDPAU_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_VDPAU_RELEASE "${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_RELEASE}"
                                  "release" vdpau)
    set(_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_VDPAU_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_VDPAU_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VDPAU_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_VDPAU_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_VDPAU_RELWITHDEBINFO "${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" vdpau)
    set(_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_VDPAU_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_VDPAU_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VDPAU_MINSIZEREL}" "${CONAN_LIB_DIRS_VDPAU_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_VDPAU_MINSIZEREL "${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" vdpau)

    add_library(CONAN_PKG::vdpau INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::vdpau PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_VDPAU} ${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VDPAU_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VDPAU_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VDPAU_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_VDPAU_RELEASE} ${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VDPAU_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VDPAU_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VDPAU_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_VDPAU_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VDPAU_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VDPAU_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VDPAU_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_VDPAU_MINSIZEREL} ${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VDPAU_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VDPAU_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VDPAU_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_VDPAU_DEBUG} ${_CONAN_PKG_LIBS_VDPAU_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VDPAU_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VDPAU_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VDPAU_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::vdpau PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_VDPAU}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_VDPAU_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_VDPAU_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_VDPAU_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_VDPAU_DEBUG}>)
    set_property(TARGET CONAN_PKG::vdpau PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_VDPAU}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_VDPAU_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_VDPAU_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_VDPAU_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_VDPAU_DEBUG}>)
    set_property(TARGET CONAN_PKG::vdpau PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_VDPAU_LIST} ${CONAN_CXX_FLAGS_VDPAU_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_VDPAU_RELEASE_LIST} ${CONAN_CXX_FLAGS_VDPAU_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_VDPAU_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_VDPAU_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_VDPAU_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_VDPAU_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_VDPAU_DEBUG_LIST}  ${CONAN_CXX_FLAGS_VDPAU_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBPNG} ${CONAN_FRAMEWORKS_FOUND_LIBPNG} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBPNG}" "${CONAN_LIB_DIRS_LIBPNG}"
                                  CONAN_PACKAGE_TARGETS_LIBPNG "${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES}"
                                  "" libpng)
    set(_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBPNG_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBPNG_DEBUG} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBPNG_DEBUG}" "${CONAN_LIB_DIRS_LIBPNG_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBPNG_DEBUG "${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_DEBUG}"
                                  "debug" libpng)
    set(_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBPNG_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBPNG_RELEASE} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBPNG_RELEASE}" "${CONAN_LIB_DIRS_LIBPNG_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBPNG_RELEASE "${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_RELEASE}"
                                  "release" libpng)
    set(_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBPNG_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBPNG_RELWITHDEBINFO} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBPNG_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBPNG_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBPNG_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libpng)
    set(_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBPNG_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBPNG_MINSIZEREL} CONAN_PKG::zlib")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBPNG_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBPNG_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBPNG_MINSIZEREL "${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libpng)

    add_library(CONAN_PKG::libpng INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libpng PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBPNG} ${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBPNG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBPNG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBPNG_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBPNG_RELEASE} ${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBPNG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBPNG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBPNG_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBPNG_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBPNG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBPNG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBPNG_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBPNG_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBPNG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBPNG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBPNG_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBPNG_DEBUG} ${_CONAN_PKG_LIBS_LIBPNG_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBPNG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBPNG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBPNG_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libpng PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBPNG}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBPNG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBPNG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBPNG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBPNG_DEBUG}>)
    set_property(TARGET CONAN_PKG::libpng PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBPNG}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBPNG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBPNG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBPNG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBPNG_DEBUG}>)
    set_property(TARGET CONAN_PKG::libpng PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBPNG_LIST} ${CONAN_CXX_FLAGS_LIBPNG_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBPNG_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBPNG_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBPNG_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBPNG_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBPNG_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBPNG_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBPNG_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBPNG_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES "${CONAN_SYSTEM_LIBS_BZIP2} ${CONAN_FRAMEWORKS_FOUND_BZIP2} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BZIP2_DEPENDENCIES "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BZIP2}" "${CONAN_LIB_DIRS_BZIP2}"
                                  CONAN_PACKAGE_TARGETS_BZIP2 "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES}"
                                  "" bzip2)
    set(_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_BZIP2_DEBUG} ${CONAN_FRAMEWORKS_FOUND_BZIP2_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BZIP2_DEBUG}" "${CONAN_LIB_DIRS_BZIP2_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_BZIP2_DEBUG "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_DEBUG}"
                                  "debug" bzip2)
    set(_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_BZIP2_RELEASE} ${CONAN_FRAMEWORKS_FOUND_BZIP2_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BZIP2_RELEASE}" "${CONAN_LIB_DIRS_BZIP2_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_BZIP2_RELEASE "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELEASE}"
                                  "release" bzip2)
    set(_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_BZIP2_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_BZIP2_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BZIP2_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_BZIP2_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_BZIP2_RELWITHDEBINFO "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" bzip2)
    set(_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_BZIP2_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_BZIP2_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BZIP2_MINSIZEREL}" "${CONAN_LIB_DIRS_BZIP2_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_BZIP2_MINSIZEREL "${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" bzip2)

    add_library(CONAN_PKG::bzip2 INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::bzip2 PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_BZIP2} ${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BZIP2_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_BZIP2_RELEASE} ${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BZIP2_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_BZIP2_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BZIP2_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_BZIP2_MINSIZEREL} ${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BZIP2_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_BZIP2_DEBUG} ${_CONAN_PKG_LIBS_BZIP2_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BZIP2_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BZIP2_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::bzip2 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_BZIP2}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_BZIP2_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_BZIP2_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_BZIP2_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_BZIP2_DEBUG}>)
    set_property(TARGET CONAN_PKG::bzip2 PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_BZIP2}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_BZIP2_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_BZIP2_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_BZIP2_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_BZIP2_DEBUG}>)
    set_property(TARGET CONAN_PKG::bzip2 PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_BZIP2_LIST} ${CONAN_CXX_FLAGS_BZIP2_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_BZIP2_RELEASE_LIST} ${CONAN_CXX_FLAGS_BZIP2_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_BZIP2_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_BZIP2_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_BZIP2_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_BZIP2_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_BZIP2_DEBUG_LIST}  ${CONAN_CXX_FLAGS_BZIP2_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBICONV} ${CONAN_FRAMEWORKS_FOUND_LIBICONV} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBICONV}" "${CONAN_LIB_DIRS_LIBICONV}"
                                  CONAN_PACKAGE_TARGETS_LIBICONV "${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES}"
                                  "" libiconv)
    set(_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBICONV_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBICONV_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBICONV_DEBUG}" "${CONAN_LIB_DIRS_LIBICONV_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBICONV_DEBUG "${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_DEBUG}"
                                  "debug" libiconv)
    set(_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBICONV_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBICONV_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBICONV_RELEASE}" "${CONAN_LIB_DIRS_LIBICONV_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBICONV_RELEASE "${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_RELEASE}"
                                  "release" libiconv)
    set(_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBICONV_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBICONV_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBICONV_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBICONV_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBICONV_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libiconv)
    set(_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBICONV_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBICONV_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBICONV_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBICONV_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBICONV_MINSIZEREL "${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libiconv)

    add_library(CONAN_PKG::libiconv INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libiconv PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBICONV} ${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBICONV_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBICONV_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBICONV_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBICONV_RELEASE} ${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBICONV_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBICONV_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBICONV_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBICONV_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBICONV_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBICONV_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBICONV_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBICONV_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBICONV_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBICONV_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBICONV_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBICONV_DEBUG} ${_CONAN_PKG_LIBS_LIBICONV_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBICONV_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBICONV_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBICONV_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libiconv PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBICONV}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBICONV_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBICONV_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBICONV_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBICONV_DEBUG}>)
    set_property(TARGET CONAN_PKG::libiconv PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBICONV}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBICONV_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBICONV_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBICONV_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBICONV_DEBUG}>)
    set_property(TARGET CONAN_PKG::libiconv PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBICONV_LIST} ${CONAN_CXX_FLAGS_LIBICONV_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBICONV_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBICONV_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBICONV_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBICONV_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBICONV_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBICONV_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBICONV_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBICONV_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES "${CONAN_SYSTEM_LIBS_BROTLI} ${CONAN_FRAMEWORKS_FOUND_BROTLI} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BROTLI_DEPENDENCIES "${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BROTLI}" "${CONAN_LIB_DIRS_BROTLI}"
                                  CONAN_PACKAGE_TARGETS_BROTLI "${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES}"
                                  "" brotli)
    set(_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_BROTLI_DEBUG} ${CONAN_FRAMEWORKS_FOUND_BROTLI_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BROTLI_DEBUG}" "${CONAN_LIB_DIRS_BROTLI_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_BROTLI_DEBUG "${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_DEBUG}"
                                  "debug" brotli)
    set(_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_BROTLI_RELEASE} ${CONAN_FRAMEWORKS_FOUND_BROTLI_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BROTLI_RELEASE}" "${CONAN_LIB_DIRS_BROTLI_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_BROTLI_RELEASE "${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_RELEASE}"
                                  "release" brotli)
    set(_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_BROTLI_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_BROTLI_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BROTLI_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_BROTLI_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_BROTLI_RELWITHDEBINFO "${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" brotli)
    set(_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_BROTLI_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_BROTLI_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_BROTLI_MINSIZEREL}" "${CONAN_LIB_DIRS_BROTLI_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_BROTLI_MINSIZEREL "${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" brotli)

    add_library(CONAN_PKG::brotli INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::brotli PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_BROTLI} ${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BROTLI_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BROTLI_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BROTLI_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_BROTLI_RELEASE} ${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BROTLI_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BROTLI_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BROTLI_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_BROTLI_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BROTLI_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BROTLI_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BROTLI_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_BROTLI_MINSIZEREL} ${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BROTLI_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BROTLI_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BROTLI_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_BROTLI_DEBUG} ${_CONAN_PKG_LIBS_BROTLI_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BROTLI_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_BROTLI_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_BROTLI_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::brotli PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_BROTLI}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_BROTLI_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_BROTLI_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_BROTLI_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_BROTLI_DEBUG}>)
    set_property(TARGET CONAN_PKG::brotli PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_BROTLI}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_BROTLI_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_BROTLI_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_BROTLI_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_BROTLI_DEBUG}>)
    set_property(TARGET CONAN_PKG::brotli PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_BROTLI_LIST} ${CONAN_CXX_FLAGS_BROTLI_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_BROTLI_RELEASE_LIST} ${CONAN_CXX_FLAGS_BROTLI_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_BROTLI_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_BROTLI_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_BROTLI_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_BROTLI_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_BROTLI_DEBUG_LIST}  ${CONAN_CXX_FLAGS_BROTLI_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES "${CONAN_SYSTEM_LIBS_OPENSSL} ${CONAN_FRAMEWORKS_FOUND_OPENSSL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES "${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENSSL}" "${CONAN_LIB_DIRS_OPENSSL}"
                                  CONAN_PACKAGE_TARGETS_OPENSSL "${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES}"
                                  "" openssl)
    set(_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_OPENSSL_DEBUG} ${CONAN_FRAMEWORKS_FOUND_OPENSSL_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENSSL_DEBUG}" "${CONAN_LIB_DIRS_OPENSSL_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_OPENSSL_DEBUG "${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_DEBUG}"
                                  "debug" openssl)
    set(_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_OPENSSL_RELEASE} ${CONAN_FRAMEWORKS_FOUND_OPENSSL_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENSSL_RELEASE}" "${CONAN_LIB_DIRS_OPENSSL_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_OPENSSL_RELEASE "${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_RELEASE}"
                                  "release" openssl)
    set(_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_OPENSSL_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_OPENSSL_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENSSL_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_OPENSSL_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_OPENSSL_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" openssl)
    set(_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_OPENSSL_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_OPENSSL_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPENSSL_MINSIZEREL}" "${CONAN_LIB_DIRS_OPENSSL_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_OPENSSL_MINSIZEREL "${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" openssl)

    add_library(CONAN_PKG::openssl INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::openssl PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_OPENSSL} ${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENSSL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENSSL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENSSL_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_OPENSSL_RELEASE} ${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENSSL_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENSSL_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENSSL_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_OPENSSL_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENSSL_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENSSL_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENSSL_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_OPENSSL_MINSIZEREL} ${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENSSL_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENSSL_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENSSL_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_OPENSSL_DEBUG} ${_CONAN_PKG_LIBS_OPENSSL_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENSSL_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPENSSL_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPENSSL_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::openssl PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_OPENSSL}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_OPENSSL_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_OPENSSL_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_OPENSSL_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_OPENSSL_DEBUG}>)
    set_property(TARGET CONAN_PKG::openssl PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_OPENSSL}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_OPENSSL_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_OPENSSL_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_OPENSSL_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_OPENSSL_DEBUG}>)
    set_property(TARGET CONAN_PKG::openssl PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_OPENSSL_LIST} ${CONAN_CXX_FLAGS_OPENSSL_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_OPENSSL_RELEASE_LIST} ${CONAN_CXX_FLAGS_OPENSSL_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_OPENSSL_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_OPENSSL_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_OPENSSL_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_OPENSSL_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_OPENSSL_DEBUG_LIST}  ${CONAN_CXX_FLAGS_OPENSSL_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBALSA} ${CONAN_FRAMEWORKS_FOUND_LIBALSA} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBALSA}" "${CONAN_LIB_DIRS_LIBALSA}"
                                  CONAN_PACKAGE_TARGETS_LIBALSA "${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES}"
                                  "" libalsa)
    set(_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBALSA_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBALSA_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBALSA_DEBUG}" "${CONAN_LIB_DIRS_LIBALSA_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBALSA_DEBUG "${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_DEBUG}"
                                  "debug" libalsa)
    set(_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBALSA_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBALSA_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBALSA_RELEASE}" "${CONAN_LIB_DIRS_LIBALSA_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBALSA_RELEASE "${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_RELEASE}"
                                  "release" libalsa)
    set(_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBALSA_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBALSA_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBALSA_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBALSA_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBALSA_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libalsa)
    set(_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBALSA_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBALSA_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBALSA_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBALSA_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBALSA_MINSIZEREL "${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libalsa)

    add_library(CONAN_PKG::libalsa INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libalsa PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBALSA} ${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBALSA_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBALSA_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBALSA_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBALSA_RELEASE} ${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBALSA_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBALSA_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBALSA_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBALSA_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBALSA_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBALSA_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBALSA_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBALSA_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBALSA_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBALSA_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBALSA_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBALSA_DEBUG} ${_CONAN_PKG_LIBS_LIBALSA_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBALSA_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBALSA_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBALSA_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libalsa PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBALSA}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBALSA_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBALSA_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBALSA_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBALSA_DEBUG}>)
    set_property(TARGET CONAN_PKG::libalsa PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBALSA}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBALSA_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBALSA_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBALSA_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBALSA_DEBUG}>)
    set_property(TARGET CONAN_PKG::libalsa PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBALSA_LIST} ${CONAN_CXX_FLAGS_LIBALSA_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBALSA_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBALSA_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBALSA_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBALSA_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBALSA_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBALSA_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBALSA_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBALSA_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_XORG_DEPENDENCIES "${CONAN_SYSTEM_LIBS_XORG} ${CONAN_FRAMEWORKS_FOUND_XORG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_XORG_DEPENDENCIES "${_CONAN_PKG_LIBS_XORG_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_XORG}" "${CONAN_LIB_DIRS_XORG}"
                                  CONAN_PACKAGE_TARGETS_XORG "${_CONAN_PKG_LIBS_XORG_DEPENDENCIES}"
                                  "" xorg)
    set(_CONAN_PKG_LIBS_XORG_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_XORG_DEBUG} ${CONAN_FRAMEWORKS_FOUND_XORG_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_XORG_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_XORG_DEBUG}" "${CONAN_LIB_DIRS_XORG_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_XORG_DEBUG "${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_DEBUG}"
                                  "debug" xorg)
    set(_CONAN_PKG_LIBS_XORG_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_XORG_RELEASE} ${CONAN_FRAMEWORKS_FOUND_XORG_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_XORG_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_XORG_RELEASE}" "${CONAN_LIB_DIRS_XORG_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_XORG_RELEASE "${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_RELEASE}"
                                  "release" xorg)
    set(_CONAN_PKG_LIBS_XORG_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_XORG_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_XORG_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_XORG_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_XORG_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_XORG_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_XORG_RELWITHDEBINFO "${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" xorg)
    set(_CONAN_PKG_LIBS_XORG_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_XORG_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_XORG_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_XORG_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_XORG_MINSIZEREL}" "${CONAN_LIB_DIRS_XORG_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_XORG_MINSIZEREL "${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" xorg)

    add_library(CONAN_PKG::xorg INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::xorg PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_XORG} ${_CONAN_PKG_LIBS_XORG_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XORG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XORG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_XORG_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_XORG_RELEASE} ${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XORG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XORG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_XORG_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_XORG_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XORG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XORG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_XORG_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_XORG_MINSIZEREL} ${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XORG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XORG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_XORG_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_XORG_DEBUG} ${_CONAN_PKG_LIBS_XORG_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XORG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_XORG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_XORG_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::xorg PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_XORG}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_XORG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_XORG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_XORG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_XORG_DEBUG}>)
    set_property(TARGET CONAN_PKG::xorg PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_XORG}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_XORG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_XORG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_XORG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_XORG_DEBUG}>)
    set_property(TARGET CONAN_PKG::xorg PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_XORG_LIST} ${CONAN_CXX_FLAGS_XORG_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_XORG_RELEASE_LIST} ${CONAN_CXX_FLAGS_XORG_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_XORG_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_XORG_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_XORG_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_XORG_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_XORG_DEBUG_LIST}  ${CONAN_CXX_FLAGS_XORG_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBSNDFILE} ${CONAN_FRAMEWORKS_FOUND_LIBSNDFILE} CONAN_PKG::ogg CONAN_PKG::vorbis CONAN_PKG::flac CONAN_PKG::opus")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBSNDFILE}" "${CONAN_LIB_DIRS_LIBSNDFILE}"
                                  CONAN_PACKAGE_TARGETS_LIBSNDFILE "${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES}"
                                  "" libsndfile)
    set(_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBSNDFILE_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBSNDFILE_DEBUG} CONAN_PKG::ogg CONAN_PKG::vorbis CONAN_PKG::flac CONAN_PKG::opus")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBSNDFILE_DEBUG}" "${CONAN_LIB_DIRS_LIBSNDFILE_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBSNDFILE_DEBUG "${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_DEBUG}"
                                  "debug" libsndfile)
    set(_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBSNDFILE_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBSNDFILE_RELEASE} CONAN_PKG::ogg CONAN_PKG::vorbis CONAN_PKG::flac CONAN_PKG::opus")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBSNDFILE_RELEASE}" "${CONAN_LIB_DIRS_LIBSNDFILE_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBSNDFILE_RELEASE "${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_RELEASE}"
                                  "release" libsndfile)
    set(_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBSNDFILE_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBSNDFILE_RELWITHDEBINFO} CONAN_PKG::ogg CONAN_PKG::vorbis CONAN_PKG::flac CONAN_PKG::opus")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBSNDFILE_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBSNDFILE_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBSNDFILE_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libsndfile)
    set(_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBSNDFILE_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBSNDFILE_MINSIZEREL} CONAN_PKG::ogg CONAN_PKG::vorbis CONAN_PKG::flac CONAN_PKG::opus")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBSNDFILE_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBSNDFILE_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBSNDFILE_MINSIZEREL "${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libsndfile)

    add_library(CONAN_PKG::libsndfile INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libsndfile PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBSNDFILE} ${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBSNDFILE_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBSNDFILE_RELEASE} ${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBSNDFILE_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBSNDFILE_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBSNDFILE_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBSNDFILE_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBSNDFILE_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBSNDFILE_DEBUG} ${_CONAN_PKG_LIBS_LIBSNDFILE_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBSNDFILE_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBSNDFILE_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libsndfile PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBSNDFILE}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBSNDFILE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBSNDFILE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBSNDFILE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBSNDFILE_DEBUG}>)
    set_property(TARGET CONAN_PKG::libsndfile PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBSNDFILE}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBSNDFILE_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBSNDFILE_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBSNDFILE_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBSNDFILE_DEBUG}>)
    set_property(TARGET CONAN_PKG::libsndfile PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBSNDFILE_LIST} ${CONAN_CXX_FLAGS_LIBSNDFILE_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBSNDFILE_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBSNDFILE_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBSNDFILE_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBSNDFILE_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBSNDFILE_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBSNDFILE_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBSNDFILE_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBSNDFILE_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES "${CONAN_SYSTEM_LIBS_LIBCAP} ${CONAN_FRAMEWORKS_FOUND_LIBCAP} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES "${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBCAP}" "${CONAN_LIB_DIRS_LIBCAP}"
                                  CONAN_PACKAGE_TARGETS_LIBCAP "${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES}"
                                  "" libcap)
    set(_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_LIBCAP_DEBUG} ${CONAN_FRAMEWORKS_FOUND_LIBCAP_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBCAP_DEBUG}" "${CONAN_LIB_DIRS_LIBCAP_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_LIBCAP_DEBUG "${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_DEBUG}"
                                  "debug" libcap)
    set(_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_LIBCAP_RELEASE} ${CONAN_FRAMEWORKS_FOUND_LIBCAP_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBCAP_RELEASE}" "${CONAN_LIB_DIRS_LIBCAP_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_LIBCAP_RELEASE "${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_RELEASE}"
                                  "release" libcap)
    set(_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_LIBCAP_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_LIBCAP_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBCAP_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_LIBCAP_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_LIBCAP_RELWITHDEBINFO "${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" libcap)
    set(_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_LIBCAP_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_LIBCAP_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_LIBCAP_MINSIZEREL}" "${CONAN_LIB_DIRS_LIBCAP_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_LIBCAP_MINSIZEREL "${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" libcap)

    add_library(CONAN_PKG::libcap INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::libcap PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_LIBCAP} ${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBCAP_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBCAP_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBCAP_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_LIBCAP_RELEASE} ${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBCAP_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBCAP_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBCAP_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_LIBCAP_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBCAP_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBCAP_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBCAP_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_LIBCAP_MINSIZEREL} ${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBCAP_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBCAP_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBCAP_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_LIBCAP_DEBUG} ${_CONAN_PKG_LIBS_LIBCAP_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBCAP_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_LIBCAP_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_LIBCAP_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::libcap PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_LIBCAP}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_LIBCAP_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_LIBCAP_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_LIBCAP_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_LIBCAP_DEBUG}>)
    set_property(TARGET CONAN_PKG::libcap PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_LIBCAP}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_LIBCAP_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_LIBCAP_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_LIBCAP_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_LIBCAP_DEBUG}>)
    set_property(TARGET CONAN_PKG::libcap PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_LIBCAP_LIST} ${CONAN_CXX_FLAGS_LIBCAP_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_LIBCAP_RELEASE_LIST} ${CONAN_CXX_FLAGS_LIBCAP_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_LIBCAP_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_LIBCAP_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_LIBCAP_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_LIBCAP_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_LIBCAP_DEBUG_LIST}  ${CONAN_CXX_FLAGS_LIBCAP_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES "${CONAN_SYSTEM_LIBS_ZLIB} ${CONAN_FRAMEWORKS_FOUND_ZLIB} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZLIB_DEPENDENCIES "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZLIB}" "${CONAN_LIB_DIRS_ZLIB}"
                                  CONAN_PACKAGE_TARGETS_ZLIB "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES}"
                                  "" zlib)
    set(_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_ZLIB_DEBUG} ${CONAN_FRAMEWORKS_FOUND_ZLIB_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZLIB_DEBUG}" "${CONAN_LIB_DIRS_ZLIB_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_ZLIB_DEBUG "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_DEBUG}"
                                  "debug" zlib)
    set(_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_ZLIB_RELEASE} ${CONAN_FRAMEWORKS_FOUND_ZLIB_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZLIB_RELEASE}" "${CONAN_LIB_DIRS_ZLIB_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_ZLIB_RELEASE "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELEASE}"
                                  "release" zlib)
    set(_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_ZLIB_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_ZLIB_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZLIB_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_ZLIB_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_ZLIB_RELWITHDEBINFO "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" zlib)
    set(_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_ZLIB_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_ZLIB_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_ZLIB_MINSIZEREL}" "${CONAN_LIB_DIRS_ZLIB_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_ZLIB_MINSIZEREL "${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" zlib)

    add_library(CONAN_PKG::zlib INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::zlib PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_ZLIB} ${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZLIB_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_ZLIB_RELEASE} ${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZLIB_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_ZLIB_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZLIB_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_ZLIB_MINSIZEREL} ${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZLIB_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_ZLIB_DEBUG} ${_CONAN_PKG_LIBS_ZLIB_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_ZLIB_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_ZLIB_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::zlib PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_ZLIB}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_ZLIB_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_ZLIB_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_ZLIB_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_ZLIB_DEBUG}>)
    set_property(TARGET CONAN_PKG::zlib PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_ZLIB}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_ZLIB_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_ZLIB_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_ZLIB_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_ZLIB_DEBUG}>)
    set_property(TARGET CONAN_PKG::zlib PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_ZLIB_LIST} ${CONAN_CXX_FLAGS_ZLIB_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_ZLIB_RELEASE_LIST} ${CONAN_CXX_FLAGS_ZLIB_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_ZLIB_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_ZLIB_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_ZLIB_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_ZLIB_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_ZLIB_DEBUG_LIST}  ${CONAN_CXX_FLAGS_ZLIB_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES "${CONAN_SYSTEM_LIBS_VORBIS} ${CONAN_FRAMEWORKS_FOUND_VORBIS} CONAN_PKG::ogg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VORBIS_DEPENDENCIES "${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VORBIS}" "${CONAN_LIB_DIRS_VORBIS}"
                                  CONAN_PACKAGE_TARGETS_VORBIS "${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES}"
                                  "" vorbis)
    set(_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_VORBIS_DEBUG} ${CONAN_FRAMEWORKS_FOUND_VORBIS_DEBUG} CONAN_PKG::ogg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VORBIS_DEBUG}" "${CONAN_LIB_DIRS_VORBIS_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_VORBIS_DEBUG "${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_DEBUG}"
                                  "debug" vorbis)
    set(_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_VORBIS_RELEASE} ${CONAN_FRAMEWORKS_FOUND_VORBIS_RELEASE} CONAN_PKG::ogg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VORBIS_RELEASE}" "${CONAN_LIB_DIRS_VORBIS_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_VORBIS_RELEASE "${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_RELEASE}"
                                  "release" vorbis)
    set(_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_VORBIS_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_VORBIS_RELWITHDEBINFO} CONAN_PKG::ogg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VORBIS_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_VORBIS_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_VORBIS_RELWITHDEBINFO "${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" vorbis)
    set(_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_VORBIS_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_VORBIS_MINSIZEREL} CONAN_PKG::ogg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_VORBIS_MINSIZEREL}" "${CONAN_LIB_DIRS_VORBIS_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_VORBIS_MINSIZEREL "${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" vorbis)

    add_library(CONAN_PKG::vorbis INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::vorbis PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_VORBIS} ${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VORBIS_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VORBIS_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VORBIS_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_VORBIS_RELEASE} ${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VORBIS_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VORBIS_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VORBIS_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_VORBIS_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VORBIS_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VORBIS_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VORBIS_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_VORBIS_MINSIZEREL} ${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VORBIS_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VORBIS_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VORBIS_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_VORBIS_DEBUG} ${_CONAN_PKG_LIBS_VORBIS_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VORBIS_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_VORBIS_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_VORBIS_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::vorbis PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_VORBIS}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_VORBIS_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_VORBIS_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_VORBIS_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_VORBIS_DEBUG}>)
    set_property(TARGET CONAN_PKG::vorbis PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_VORBIS}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_VORBIS_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_VORBIS_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_VORBIS_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_VORBIS_DEBUG}>)
    set_property(TARGET CONAN_PKG::vorbis PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_VORBIS_LIST} ${CONAN_CXX_FLAGS_VORBIS_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_VORBIS_RELEASE_LIST} ${CONAN_CXX_FLAGS_VORBIS_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_VORBIS_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_VORBIS_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_VORBIS_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_VORBIS_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_VORBIS_DEBUG_LIST}  ${CONAN_CXX_FLAGS_VORBIS_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_OPUS_DEPENDENCIES "${CONAN_SYSTEM_LIBS_OPUS} ${CONAN_FRAMEWORKS_FOUND_OPUS} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPUS_DEPENDENCIES "${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPUS}" "${CONAN_LIB_DIRS_OPUS}"
                                  CONAN_PACKAGE_TARGETS_OPUS "${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES}"
                                  "" opus)
    set(_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_OPUS_DEBUG} ${CONAN_FRAMEWORKS_FOUND_OPUS_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPUS_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPUS_DEBUG}" "${CONAN_LIB_DIRS_OPUS_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_OPUS_DEBUG "${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_DEBUG}"
                                  "debug" opus)
    set(_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_OPUS_RELEASE} ${CONAN_FRAMEWORKS_FOUND_OPUS_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPUS_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPUS_RELEASE}" "${CONAN_LIB_DIRS_OPUS_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_OPUS_RELEASE "${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_RELEASE}"
                                  "release" opus)
    set(_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_OPUS_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_OPUS_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPUS_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPUS_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_OPUS_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_OPUS_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" opus)
    set(_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_OPUS_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_OPUS_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OPUS_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OPUS_MINSIZEREL}" "${CONAN_LIB_DIRS_OPUS_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_OPUS_MINSIZEREL "${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" opus)

    add_library(CONAN_PKG::opus INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::opus PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_OPUS} ${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPUS_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPUS_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPUS_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_OPUS_RELEASE} ${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPUS_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPUS_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPUS_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_OPUS_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPUS_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPUS_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPUS_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_OPUS_MINSIZEREL} ${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPUS_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPUS_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPUS_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_OPUS_DEBUG} ${_CONAN_PKG_LIBS_OPUS_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPUS_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OPUS_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OPUS_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::opus PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_OPUS}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_OPUS_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_OPUS_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_OPUS_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_OPUS_DEBUG}>)
    set_property(TARGET CONAN_PKG::opus PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_OPUS}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_OPUS_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_OPUS_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_OPUS_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_OPUS_DEBUG}>)
    set_property(TARGET CONAN_PKG::opus PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_OPUS_LIST} ${CONAN_CXX_FLAGS_OPUS_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_OPUS_RELEASE_LIST} ${CONAN_CXX_FLAGS_OPUS_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_OPUS_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_OPUS_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_OPUS_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_OPUS_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_OPUS_DEBUG_LIST}  ${CONAN_CXX_FLAGS_OPUS_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_FLAC_DEPENDENCIES "${CONAN_SYSTEM_LIBS_FLAC} ${CONAN_FRAMEWORKS_FOUND_FLAC} CONAN_PKG::ogg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FLAC_DEPENDENCIES "${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FLAC}" "${CONAN_LIB_DIRS_FLAC}"
                                  CONAN_PACKAGE_TARGETS_FLAC "${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES}"
                                  "" flac)
    set(_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_FLAC_DEBUG} ${CONAN_FRAMEWORKS_FOUND_FLAC_DEBUG} CONAN_PKG::ogg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FLAC_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FLAC_DEBUG}" "${CONAN_LIB_DIRS_FLAC_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_FLAC_DEBUG "${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_DEBUG}"
                                  "debug" flac)
    set(_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_FLAC_RELEASE} ${CONAN_FRAMEWORKS_FOUND_FLAC_RELEASE} CONAN_PKG::ogg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FLAC_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FLAC_RELEASE}" "${CONAN_LIB_DIRS_FLAC_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_FLAC_RELEASE "${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_RELEASE}"
                                  "release" flac)
    set(_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_FLAC_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_FLAC_RELWITHDEBINFO} CONAN_PKG::ogg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FLAC_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FLAC_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_FLAC_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_FLAC_RELWITHDEBINFO "${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" flac)
    set(_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_FLAC_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_FLAC_MINSIZEREL} CONAN_PKG::ogg")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_FLAC_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_FLAC_MINSIZEREL}" "${CONAN_LIB_DIRS_FLAC_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_FLAC_MINSIZEREL "${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" flac)

    add_library(CONAN_PKG::flac INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::flac PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_FLAC} ${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FLAC_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FLAC_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FLAC_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_FLAC_RELEASE} ${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FLAC_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FLAC_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FLAC_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_FLAC_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FLAC_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FLAC_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FLAC_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_FLAC_MINSIZEREL} ${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FLAC_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FLAC_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FLAC_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_FLAC_DEBUG} ${_CONAN_PKG_LIBS_FLAC_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FLAC_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_FLAC_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_FLAC_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::flac PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_FLAC}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_FLAC_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_FLAC_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_FLAC_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_FLAC_DEBUG}>)
    set_property(TARGET CONAN_PKG::flac PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_FLAC}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_FLAC_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_FLAC_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_FLAC_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_FLAC_DEBUG}>)
    set_property(TARGET CONAN_PKG::flac PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_FLAC_LIST} ${CONAN_CXX_FLAGS_FLAC_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_FLAC_RELEASE_LIST} ${CONAN_CXX_FLAGS_FLAC_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_FLAC_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_FLAC_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_FLAC_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_FLAC_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_FLAC_DEBUG_LIST}  ${CONAN_CXX_FLAGS_FLAC_DEBUG_LIST}>)


    set(_CONAN_PKG_LIBS_OGG_DEPENDENCIES "${CONAN_SYSTEM_LIBS_OGG} ${CONAN_FRAMEWORKS_FOUND_OGG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OGG_DEPENDENCIES "${_CONAN_PKG_LIBS_OGG_DEPENDENCIES}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OGG}" "${CONAN_LIB_DIRS_OGG}"
                                  CONAN_PACKAGE_TARGETS_OGG "${_CONAN_PKG_LIBS_OGG_DEPENDENCIES}"
                                  "" ogg)
    set(_CONAN_PKG_LIBS_OGG_DEPENDENCIES_DEBUG "${CONAN_SYSTEM_LIBS_OGG_DEBUG} ${CONAN_FRAMEWORKS_FOUND_OGG_DEBUG} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OGG_DEPENDENCIES_DEBUG "${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_DEBUG}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OGG_DEBUG}" "${CONAN_LIB_DIRS_OGG_DEBUG}"
                                  CONAN_PACKAGE_TARGETS_OGG_DEBUG "${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_DEBUG}"
                                  "debug" ogg)
    set(_CONAN_PKG_LIBS_OGG_DEPENDENCIES_RELEASE "${CONAN_SYSTEM_LIBS_OGG_RELEASE} ${CONAN_FRAMEWORKS_FOUND_OGG_RELEASE} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OGG_DEPENDENCIES_RELEASE "${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_RELEASE}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OGG_RELEASE}" "${CONAN_LIB_DIRS_OGG_RELEASE}"
                                  CONAN_PACKAGE_TARGETS_OGG_RELEASE "${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_RELEASE}"
                                  "release" ogg)
    set(_CONAN_PKG_LIBS_OGG_DEPENDENCIES_RELWITHDEBINFO "${CONAN_SYSTEM_LIBS_OGG_RELWITHDEBINFO} ${CONAN_FRAMEWORKS_FOUND_OGG_RELWITHDEBINFO} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OGG_DEPENDENCIES_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_RELWITHDEBINFO}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OGG_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_OGG_RELWITHDEBINFO}"
                                  CONAN_PACKAGE_TARGETS_OGG_RELWITHDEBINFO "${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_RELWITHDEBINFO}"
                                  "relwithdebinfo" ogg)
    set(_CONAN_PKG_LIBS_OGG_DEPENDENCIES_MINSIZEREL "${CONAN_SYSTEM_LIBS_OGG_MINSIZEREL} ${CONAN_FRAMEWORKS_FOUND_OGG_MINSIZEREL} ")
    string(REPLACE " " ";" _CONAN_PKG_LIBS_OGG_DEPENDENCIES_MINSIZEREL "${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_MINSIZEREL}")
    conan_package_library_targets("${CONAN_PKG_LIBS_OGG_MINSIZEREL}" "${CONAN_LIB_DIRS_OGG_MINSIZEREL}"
                                  CONAN_PACKAGE_TARGETS_OGG_MINSIZEREL "${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_MINSIZEREL}"
                                  "minsizerel" ogg)

    add_library(CONAN_PKG::ogg INTERFACE IMPORTED)

    # Property INTERFACE_LINK_FLAGS do not work, necessary to add to INTERFACE_LINK_LIBRARIES
    set_property(TARGET CONAN_PKG::ogg PROPERTY INTERFACE_LINK_LIBRARIES ${CONAN_PACKAGE_TARGETS_OGG} ${_CONAN_PKG_LIBS_OGG_DEPENDENCIES}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OGG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OGG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OGG_LIST}>

                                                                 $<$<CONFIG:Release>:${CONAN_PACKAGE_TARGETS_OGG_RELEASE} ${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_RELEASE}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OGG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OGG_RELEASE_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OGG_RELEASE_LIST}>>

                                                                 $<$<CONFIG:RelWithDebInfo>:${CONAN_PACKAGE_TARGETS_OGG_RELWITHDEBINFO} ${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_RELWITHDEBINFO}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OGG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OGG_RELWITHDEBINFO_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OGG_RELWITHDEBINFO_LIST}>>

                                                                 $<$<CONFIG:MinSizeRel>:${CONAN_PACKAGE_TARGETS_OGG_MINSIZEREL} ${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_MINSIZEREL}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OGG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OGG_MINSIZEREL_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OGG_MINSIZEREL_LIST}>>

                                                                 $<$<CONFIG:Debug>:${CONAN_PACKAGE_TARGETS_OGG_DEBUG} ${_CONAN_PKG_LIBS_OGG_DEPENDENCIES_DEBUG}
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OGG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${CONAN_SHARED_LINKER_FLAGS_OGG_DEBUG_LIST}>
                                                                 $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${CONAN_EXE_LINKER_FLAGS_OGG_DEBUG_LIST}>>)
    set_property(TARGET CONAN_PKG::ogg PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONAN_INCLUDE_DIRS_OGG}
                                                                      $<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_OGG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_OGG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_OGG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_OGG_DEBUG}>)
    set_property(TARGET CONAN_PKG::ogg PROPERTY INTERFACE_COMPILE_DEFINITIONS ${CONAN_COMPILE_DEFINITIONS_OGG}
                                                                      $<$<CONFIG:Release>:${CONAN_COMPILE_DEFINITIONS_OGG_RELEASE}>
                                                                      $<$<CONFIG:RelWithDebInfo>:${CONAN_COMPILE_DEFINITIONS_OGG_RELWITHDEBINFO}>
                                                                      $<$<CONFIG:MinSizeRel>:${CONAN_COMPILE_DEFINITIONS_OGG_MINSIZEREL}>
                                                                      $<$<CONFIG:Debug>:${CONAN_COMPILE_DEFINITIONS_OGG_DEBUG}>)
    set_property(TARGET CONAN_PKG::ogg PROPERTY INTERFACE_COMPILE_OPTIONS ${CONAN_C_FLAGS_OGG_LIST} ${CONAN_CXX_FLAGS_OGG_LIST}
                                                                  $<$<CONFIG:Release>:${CONAN_C_FLAGS_OGG_RELEASE_LIST} ${CONAN_CXX_FLAGS_OGG_RELEASE_LIST}>
                                                                  $<$<CONFIG:RelWithDebInfo>:${CONAN_C_FLAGS_OGG_RELWITHDEBINFO_LIST} ${CONAN_CXX_FLAGS_OGG_RELWITHDEBINFO_LIST}>
                                                                  $<$<CONFIG:MinSizeRel>:${CONAN_C_FLAGS_OGG_MINSIZEREL_LIST} ${CONAN_CXX_FLAGS_OGG_MINSIZEREL_LIST}>
                                                                  $<$<CONFIG:Debug>:${CONAN_C_FLAGS_OGG_DEBUG_LIST}  ${CONAN_CXX_FLAGS_OGG_DEBUG_LIST}>)

    set(CONAN_TARGETS CONAN_PKG::opencv CONAN_PKG::jasper CONAN_PKG::openexr CONAN_PKG::libtiff CONAN_PKG::eigen CONAN_PKG::ffmpeg CONAN_PKG::quirc CONAN_PKG::gtk CONAN_PKG::protobuf CONAN_PKG::ade CONAN_PKG::libjpeg CONAN_PKG::libdeflate CONAN_PKG::xz_utils CONAN_PKG::jbig CONAN_PKG::zstd CONAN_PKG::libwebp CONAN_PKG::freetype CONAN_PKG::openjpeg CONAN_PKG::openh264 CONAN_PKG::libx264 CONAN_PKG::libx265 CONAN_PKG::libvpx CONAN_PKG::libmp3lame CONAN_PKG::libfdk_aac CONAN_PKG::pulseaudio CONAN_PKG::vaapi CONAN_PKG::vdpau CONAN_PKG::libpng CONAN_PKG::bzip2 CONAN_PKG::libiconv CONAN_PKG::brotli CONAN_PKG::openssl CONAN_PKG::libalsa CONAN_PKG::xorg CONAN_PKG::libsndfile CONAN_PKG::libcap CONAN_PKG::zlib CONAN_PKG::vorbis CONAN_PKG::opus CONAN_PKG::flac CONAN_PKG::ogg)

endmacro()


macro(conan_basic_setup)
    set(options TARGETS NO_OUTPUT_DIRS SKIP_RPATH KEEP_RPATHS SKIP_STD SKIP_FPIC)
    cmake_parse_arguments(ARGUMENTS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(CONAN_EXPORTED)
        conan_message(STATUS "Conan: called by CMake conan helper")
    endif()

    if(CONAN_IN_LOCAL_CACHE)
        conan_message(STATUS "Conan: called inside local cache")
    endif()

    if(NOT ARGUMENTS_NO_OUTPUT_DIRS)
        conan_message(STATUS "Conan: Adjusting output directories")
        conan_output_dirs_setup()
    endif()

    if(NOT ARGUMENTS_TARGETS)
        conan_message(STATUS "Conan: Using cmake global configuration")
        conan_global_flags()
    else()
        conan_message(STATUS "Conan: Using cmake targets configuration")
        conan_define_targets()
    endif()

    if(ARGUMENTS_SKIP_RPATH)
        # Change by "DEPRECATION" or "SEND_ERROR" when we are ready
        conan_message(WARNING "Conan: SKIP_RPATH is deprecated, it has been renamed to KEEP_RPATHS")
    endif()

    if(NOT ARGUMENTS_SKIP_RPATH AND NOT ARGUMENTS_KEEP_RPATHS)
        # Parameter has renamed, but we keep the compatibility with old SKIP_RPATH
        conan_set_rpath()
    endif()

    if(NOT ARGUMENTS_SKIP_STD)
        conan_set_std()
    endif()

    if(NOT ARGUMENTS_SKIP_FPIC)
        conan_set_fpic()
    endif()

    conan_check_compiler()
    conan_set_libcxx()
    conan_set_vs_runtime()
    conan_set_find_paths()
    conan_include_build_modules()
    conan_set_find_library_paths()
endmacro()


macro(conan_set_find_paths)
    # CMAKE_MODULE_PATH does not have Debug/Release config, but there are variables
    # CONAN_CMAKE_MODULE_PATH_DEBUG to be used by the consumer
    # CMake can find findXXX.cmake files in the root of packages
    set(CMAKE_MODULE_PATH ${CONAN_CMAKE_MODULE_PATH} ${CMAKE_MODULE_PATH})

    # Make find_package() to work
    set(CMAKE_PREFIX_PATH ${CONAN_CMAKE_MODULE_PATH} ${CMAKE_PREFIX_PATH})

    # Set the find root path (cross build)
    set(CMAKE_FIND_ROOT_PATH ${CONAN_CMAKE_FIND_ROOT_PATH} ${CMAKE_FIND_ROOT_PATH})
    if(CONAN_CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
        set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ${CONAN_CMAKE_FIND_ROOT_PATH_MODE_PROGRAM})
    endif()
    if(CONAN_CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
        set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ${CONAN_CMAKE_FIND_ROOT_PATH_MODE_LIBRARY})
    endif()
    if(CONAN_CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
        set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ${CONAN_CMAKE_FIND_ROOT_PATH_MODE_INCLUDE})
    endif()
endmacro()


macro(conan_set_find_library_paths)
    # CMAKE_INCLUDE_PATH, CMAKE_LIBRARY_PATH does not have Debug/Release config, but there are variables
    # CONAN_INCLUDE_DIRS_DEBUG/RELEASE CONAN_LIB_DIRS_DEBUG/RELEASE to be used by the consumer
    # For find_library
    set(CMAKE_INCLUDE_PATH ${CONAN_INCLUDE_DIRS} ${CMAKE_INCLUDE_PATH})
    set(CMAKE_LIBRARY_PATH ${CONAN_LIB_DIRS} ${CMAKE_LIBRARY_PATH})
endmacro()


macro(conan_set_vs_runtime)
    if(CONAN_LINK_RUNTIME)
        conan_get_policy(CMP0091 policy_0091)
        if(policy_0091 STREQUAL "NEW")
            if(CONAN_LINK_RUNTIME MATCHES "MTd")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug")
            elseif(CONAN_LINK_RUNTIME MATCHES "MDd")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
            elseif(CONAN_LINK_RUNTIME MATCHES "MT")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
            elseif(CONAN_LINK_RUNTIME MATCHES "MD")
                set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
            endif()
        else()
            foreach(flag CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE
                         CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_RELWITHDEBINFO
                         CMAKE_C_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_MINSIZEREL)
                if(DEFINED ${flag})
                    string(REPLACE "/MD" ${CONAN_LINK_RUNTIME} ${flag} "${${flag}}")
                endif()
            endforeach()
            foreach(flag CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG)
                if(DEFINED ${flag})
                    string(REPLACE "/MDd" ${CONAN_LINK_RUNTIME} ${flag} "${${flag}}")
                endif()
            endforeach()
        endif()
    endif()
endmacro()


macro(conan_flags_setup)
    # Macro maintained for backwards compatibility
    conan_set_find_library_paths()
    conan_global_flags()
    conan_set_rpath()
    conan_set_vs_runtime()
    conan_set_libcxx()
endmacro()


function(conan_message MESSAGE_OUTPUT)
    if(NOT CONAN_CMAKE_SILENT_OUTPUT)
        message(${ARGV${0}})
    endif()
endfunction()


function(conan_get_policy policy_id policy)
    if(POLICY "${policy_id}")
        cmake_policy(GET "${policy_id}" _policy)
        set(${policy} "${_policy}" PARENT_SCOPE)
    else()
        set(${policy} "" PARENT_SCOPE)
    endif()
endfunction()


function(conan_find_libraries_abs_path libraries package_libdir libraries_abs_path)
    foreach(_LIBRARY_NAME ${libraries})
        find_library(CONAN_FOUND_LIBRARY NAME ${_LIBRARY_NAME} PATHS ${package_libdir}
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        if(CONAN_FOUND_LIBRARY)
            conan_message(STATUS "Library ${_LIBRARY_NAME} found ${CONAN_FOUND_LIBRARY}")
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${CONAN_FOUND_LIBRARY})
        else()
            conan_message(STATUS "Library ${_LIBRARY_NAME} not found in package, might be system one")
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${_LIBRARY_NAME})
        endif()
        unset(CONAN_FOUND_LIBRARY CACHE)
    endforeach()
    set(${libraries_abs_path} ${CONAN_FULLPATH_LIBS} PARENT_SCOPE)
endfunction()


function(conan_package_library_targets libraries package_libdir libraries_abs_path deps build_type package_name)
    unset(_CONAN_ACTUAL_TARGETS CACHE)
    unset(_CONAN_FOUND_SYSTEM_LIBS CACHE)
    foreach(_LIBRARY_NAME ${libraries})
        find_library(CONAN_FOUND_LIBRARY NAME ${_LIBRARY_NAME} PATHS ${package_libdir}
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        if(CONAN_FOUND_LIBRARY)
            conan_message(STATUS "Library ${_LIBRARY_NAME} found ${CONAN_FOUND_LIBRARY}")
            set(_LIB_NAME CONAN_LIB::${package_name}_${_LIBRARY_NAME}${build_type})
            add_library(${_LIB_NAME} UNKNOWN IMPORTED)
            set_target_properties(${_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${CONAN_FOUND_LIBRARY})
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${_LIB_NAME})
            set(_CONAN_ACTUAL_TARGETS ${_CONAN_ACTUAL_TARGETS} ${_LIB_NAME})
        else()
            conan_message(STATUS "Library ${_LIBRARY_NAME} not found in package, might be system one")
            set(CONAN_FULLPATH_LIBS ${CONAN_FULLPATH_LIBS} ${_LIBRARY_NAME})
            set(_CONAN_FOUND_SYSTEM_LIBS "${_CONAN_FOUND_SYSTEM_LIBS};${_LIBRARY_NAME}")
        endif()
        unset(CONAN_FOUND_LIBRARY CACHE)
    endforeach()

    # Add all dependencies to all targets
    string(REPLACE " " ";" deps_list "${deps}")
    foreach(_CONAN_ACTUAL_TARGET ${_CONAN_ACTUAL_TARGETS})
        set_property(TARGET ${_CONAN_ACTUAL_TARGET} PROPERTY INTERFACE_LINK_LIBRARIES "${_CONAN_FOUND_SYSTEM_LIBS};${deps_list}")
    endforeach()

    set(${libraries_abs_path} ${CONAN_FULLPATH_LIBS} PARENT_SCOPE)
endfunction()


macro(conan_set_libcxx)
    if(DEFINED CONAN_LIBCXX)
        conan_message(STATUS "Conan: C++ stdlib: ${CONAN_LIBCXX}")
        if(CONAN_COMPILER STREQUAL "clang" OR CONAN_COMPILER STREQUAL "apple-clang")
            if(CONAN_LIBCXX STREQUAL "libstdc++" OR CONAN_LIBCXX STREQUAL "libstdc++11" )
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
            elseif(CONAN_LIBCXX STREQUAL "libc++")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
            endif()
        endif()
        if(CONAN_COMPILER STREQUAL "sun-cc")
            if(CONAN_LIBCXX STREQUAL "libCstd")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=Cstd")
            elseif(CONAN_LIBCXX STREQUAL "libstdcxx")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=stdcxx4")
            elseif(CONAN_LIBCXX STREQUAL "libstlport")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=stlport4")
            elseif(CONAN_LIBCXX STREQUAL "libstdc++")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -library=stdcpp")
            endif()
        endif()
        if(CONAN_LIBCXX STREQUAL "libstdc++11")
            add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
        elseif(CONAN_LIBCXX STREQUAL "libstdc++")
            add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        endif()
    endif()
endmacro()


macro(conan_set_std)
    conan_message(STATUS "Conan: Adjusting language standard")
    # Do not warn "Manually-specified variables were not used by the project"
    set(ignorevar "${CONAN_STD_CXX_FLAG}${CONAN_CMAKE_CXX_STANDARD}${CONAN_CMAKE_CXX_EXTENSIONS}")
    if (CMAKE_VERSION VERSION_LESS "3.1" OR
        (CMAKE_VERSION VERSION_LESS "3.12" AND ("${CONAN_CMAKE_CXX_STANDARD}" STREQUAL "20" OR "${CONAN_CMAKE_CXX_STANDARD}" STREQUAL "gnu20")))
        if(CONAN_STD_CXX_FLAG)
            conan_message(STATUS "Conan setting CXX_FLAGS flags: ${CONAN_STD_CXX_FLAG}")
            set(CMAKE_CXX_FLAGS "${CONAN_STD_CXX_FLAG} ${CMAKE_CXX_FLAGS}")
        endif()
    else()
        if(CONAN_CMAKE_CXX_STANDARD)
            conan_message(STATUS "Conan setting CPP STANDARD: ${CONAN_CMAKE_CXX_STANDARD} WITH EXTENSIONS ${CONAN_CMAKE_CXX_EXTENSIONS}")
            set(CMAKE_CXX_STANDARD ${CONAN_CMAKE_CXX_STANDARD})
            set(CMAKE_CXX_EXTENSIONS ${CONAN_CMAKE_CXX_EXTENSIONS})
        endif()
    endif()
endmacro()


macro(conan_set_rpath)
    conan_message(STATUS "Conan: Adjusting default RPATHs Conan policies")
    if(APPLE)
        # https://cmake.org/Wiki/CMake_RPATH_handling
        # CONAN GUIDE: All generated libraries should have the id and dependencies to other
        # dylibs without path, just the name, EX:
        # libMyLib1.dylib:
        #     libMyLib1.dylib (compatibility version 0.0.0, current version 0.0.0)
        #     libMyLib0.dylib (compatibility version 0.0.0, current version 0.0.0)
        #     /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        #     /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)
        # AVOID RPATH FOR *.dylib, ALL LIBS BETWEEN THEM AND THE EXE
        # SHOULD BE ON THE LINKER RESOLVER PATH (./ IS ONE OF THEM)
        set(CMAKE_SKIP_RPATH 1 CACHE BOOL "rpaths" FORCE)
        # Policy CMP0068
        # We want the old behavior, in CMake >= 3.9 CMAKE_SKIP_RPATH won't affect the install_name in OSX
        set(CMAKE_INSTALL_NAME_DIR "")
    endif()
endmacro()


macro(conan_set_fpic)
    if(DEFINED CONAN_CMAKE_POSITION_INDEPENDENT_CODE)
        conan_message(STATUS "Conan: Adjusting fPIC flag (${CONAN_CMAKE_POSITION_INDEPENDENT_CODE})")
        set(CMAKE_POSITION_INDEPENDENT_CODE ${CONAN_CMAKE_POSITION_INDEPENDENT_CODE})
    endif()
endmacro()


macro(conan_output_dirs_setup)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endmacro()


macro(conan_split_version VERSION_STRING MAJOR MINOR)
    #make a list from the version string
    string(REPLACE "." ";" VERSION_LIST "${VERSION_STRING}")

    #write output values
    list(LENGTH VERSION_LIST _version_len)
    list(GET VERSION_LIST 0 ${MAJOR})
    if(${_version_len} GREATER 1)
        list(GET VERSION_LIST 1 ${MINOR})
    endif()
endmacro()


macro(conan_error_compiler_version)
    message(FATAL_ERROR "Detected a mismatch for the compiler version between your conan profile settings and CMake: \n"
                        "Compiler version specified in your conan profile: ${CONAN_COMPILER_VERSION}\n"
                        "Compiler version detected in CMake: ${VERSION_MAJOR}.${VERSION_MINOR}\n"
                        "Please check your conan profile settings (conan profile show [default|your_profile_name])\n"
                        "P.S. You may set CONAN_DISABLE_CHECK_COMPILER CMake variable in order to disable this check."
           )
endmacro()

set(_CONAN_CURRENT_DIR ${CMAKE_CURRENT_LIST_DIR})

function(conan_get_compiler CONAN_INFO_COMPILER CONAN_INFO_COMPILER_VERSION)
    conan_message(STATUS "Current conanbuildinfo.cmake directory: " ${_CONAN_CURRENT_DIR})
    if(NOT EXISTS ${_CONAN_CURRENT_DIR}/conaninfo.txt)
        conan_message(STATUS "WARN: conaninfo.txt not found")
        return()
    endif()

    file (READ "${_CONAN_CURRENT_DIR}/conaninfo.txt" CONANINFO)

    # MATCHALL will match all, including the last one, which is the full_settings one
    string(REGEX MATCH "full_settings.*" _FULL_SETTINGS_MATCHED ${CONANINFO})
    string(REGEX MATCH "compiler=([-A-Za-z0-9_ ]+)" _MATCHED ${_FULL_SETTINGS_MATCHED})
    if(DEFINED CMAKE_MATCH_1)
        string(STRIP "${CMAKE_MATCH_1}" _CONAN_INFO_COMPILER)
        set(${CONAN_INFO_COMPILER} ${_CONAN_INFO_COMPILER} PARENT_SCOPE)
    endif()

    string(REGEX MATCH "compiler.version=([-A-Za-z0-9_.]+)" _MATCHED ${_FULL_SETTINGS_MATCHED})
    if(DEFINED CMAKE_MATCH_1)
        string(STRIP "${CMAKE_MATCH_1}" _CONAN_INFO_COMPILER_VERSION)
        set(${CONAN_INFO_COMPILER_VERSION} ${_CONAN_INFO_COMPILER_VERSION} PARENT_SCOPE)
    endif()
endfunction()


function(check_compiler_version)
    conan_split_version(${CMAKE_CXX_COMPILER_VERSION} VERSION_MAJOR VERSION_MINOR)
    if(DEFINED CONAN_SETTINGS_COMPILER_TOOLSET)
       conan_message(STATUS "Conan: Skipping compiler check: Declared 'compiler.toolset'")
       return()
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        # MSVC_VERSION is defined since 2.8.2 at least
        # https://cmake.org/cmake/help/v2.8.2/cmake.html#variable:MSVC_VERSION
        # https://cmake.org/cmake/help/v3.14/variable/MSVC_VERSION.html
        if(
            # 1930 = VS 17.0 (v143 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "17" AND NOT MSVC_VERSION EQUAL 1930) OR
            # 1920-1929 = VS 16.0 (v142 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "16" AND NOT((MSVC_VERSION GREATER 1919) AND (MSVC_VERSION LESS 1930))) OR
            # 1910-1919 = VS 15.0 (v141 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "15" AND NOT((MSVC_VERSION GREATER 1909) AND (MSVC_VERSION LESS 1920))) OR
            # 1900      = VS 14.0 (v140 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "14" AND NOT(MSVC_VERSION EQUAL 1900)) OR
            # 1800      = VS 12.0 (v120 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "12" AND NOT VERSION_MAJOR STREQUAL "18") OR
            # 1700      = VS 11.0 (v110 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "11" AND NOT VERSION_MAJOR STREQUAL "17") OR
            # 1600      = VS 10.0 (v100 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "10" AND NOT VERSION_MAJOR STREQUAL "16") OR
            # 1500      = VS  9.0 (v90 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "9" AND NOT VERSION_MAJOR STREQUAL "15") OR
            # 1400      = VS  8.0 (v80 toolset)
            (CONAN_COMPILER_VERSION STREQUAL "8" AND NOT VERSION_MAJOR STREQUAL "14") OR
            # 1310      = VS  7.1, 1300      = VS  7.0
            (CONAN_COMPILER_VERSION STREQUAL "7" AND NOT VERSION_MAJOR STREQUAL "13") OR
            # 1200      = VS  6.0
            (CONAN_COMPILER_VERSION STREQUAL "6" AND NOT VERSION_MAJOR STREQUAL "12") )
            conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "gcc")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        set(_CHECK_VERSION ${VERSION_MAJOR}.${VERSION_MINOR})
        set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
        if(NOT ${CONAN_COMPILER_VERSION} VERSION_LESS 5.0)
            conan_message(STATUS "Conan: Compiler GCC>=5, checking major version ${CONAN_COMPILER_VERSION}")
            conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
            if("${CONAN_COMPILER_MINOR}" STREQUAL "")
                set(_CHECK_VERSION ${VERSION_MAJOR})
                set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR})
            endif()
        endif()
        conan_message(STATUS "Conan: Checking correct version: ${_CHECK_VERSION}")
        if(NOT ${_CHECK_VERSION} VERSION_EQUAL ${_CONAN_VERSION})
            conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "clang")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        set(_CHECK_VERSION ${VERSION_MAJOR}.${VERSION_MINOR})
        set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
        if(NOT ${CONAN_COMPILER_VERSION} VERSION_LESS 8.0)
            conan_message(STATUS "Conan: Compiler Clang>=8, checking major version ${CONAN_COMPILER_VERSION}")
            if("${CONAN_COMPILER_MINOR}" STREQUAL "")
                set(_CHECK_VERSION ${VERSION_MAJOR})
                set(_CONAN_VERSION ${CONAN_COMPILER_MAJOR})
            endif()
        endif()
        conan_message(STATUS "Conan: Checking correct version: ${_CHECK_VERSION}")
        if(NOT ${_CHECK_VERSION} VERSION_EQUAL ${_CONAN_VERSION})
            conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "apple-clang" OR CONAN_COMPILER STREQUAL "sun-cc" OR CONAN_COMPILER STREQUAL "mcst-lcc")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        if(NOT ${VERSION_MAJOR}.${VERSION_MINOR} VERSION_EQUAL ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
           conan_error_compiler_version()
        endif()
    elseif(CONAN_COMPILER STREQUAL "intel")
        conan_split_version(${CONAN_COMPILER_VERSION} CONAN_COMPILER_MAJOR CONAN_COMPILER_MINOR)
        if(NOT ${CONAN_COMPILER_VERSION} VERSION_LESS 19.1)
            if(NOT ${VERSION_MAJOR}.${VERSION_MINOR} VERSION_EQUAL ${CONAN_COMPILER_MAJOR}.${CONAN_COMPILER_MINOR})
               conan_error_compiler_version()
            endif()
        else()
            if(NOT ${VERSION_MAJOR} VERSION_EQUAL ${CONAN_COMPILER_MAJOR})
               conan_error_compiler_version()
            endif()
        endif()
    else()
        conan_message(STATUS "WARN: Unknown compiler '${CONAN_COMPILER}', skipping the version check...")
    endif()
endfunction()


function(conan_check_compiler)
    if(CONAN_DISABLE_CHECK_COMPILER)
        conan_message(STATUS "WARN: Disabled conan compiler checks")
        return()
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER_ID)
        if(DEFINED CMAKE_C_COMPILER_ID)
            conan_message(STATUS "This project seems to be plain C, using '${CMAKE_C_COMPILER_ID}' compiler")
            set(CMAKE_CXX_COMPILER_ID ${CMAKE_C_COMPILER_ID})
            set(CMAKE_CXX_COMPILER_VERSION ${CMAKE_C_COMPILER_VERSION})
        else()
            message(FATAL_ERROR "This project seems to be plain C, but no compiler defined")
        endif()
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID AND NOT CMAKE_C_COMPILER_ID)
        # This use case happens when compiler is not identified by CMake, but the compilers are there and work
        conan_message(STATUS "*** WARN: CMake was not able to identify a C or C++ compiler ***")
        conan_message(STATUS "*** WARN: Disabling compiler checks. Please make sure your settings match your environment ***")
        return()
    endif()
    if(NOT DEFINED CONAN_COMPILER)
        conan_get_compiler(CONAN_COMPILER CONAN_COMPILER_VERSION)
        if(NOT DEFINED CONAN_COMPILER)
            conan_message(STATUS "WARN: CONAN_COMPILER variable not set, please make sure yourself that "
                          "your compiler and version matches your declared settings")
            return()
        endif()
    endif()

    if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL ${CMAKE_SYSTEM_NAME})
        set(CROSS_BUILDING 1)
    endif()

    # If using VS, verify toolset
    if (CONAN_COMPILER STREQUAL "Visual Studio")
        if (CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "LLVM" OR
            CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "llvm" OR
            CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "clang" OR
            CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "Clang")
            set(EXPECTED_CMAKE_CXX_COMPILER_ID "Clang")
        elseif (CONAN_SETTINGS_COMPILER_TOOLSET MATCHES "Intel")
            set(EXPECTED_CMAKE_CXX_COMPILER_ID "Intel")
        else()
            set(EXPECTED_CMAKE_CXX_COMPILER_ID "MSVC")
        endif()

        if (NOT CMAKE_CXX_COMPILER_ID MATCHES ${EXPECTED_CMAKE_CXX_COMPILER_ID})
            message(FATAL_ERROR "Incorrect '${CONAN_COMPILER}'. Toolset specifies compiler as '${EXPECTED_CMAKE_CXX_COMPILER_ID}' "
                                "but CMake detected '${CMAKE_CXX_COMPILER_ID}'")
        endif()

    # Avoid checks when cross compiling, apple-clang crashes because its APPLE but not apple-clang
    # Actually CMake is detecting "clang" when you are using apple-clang, only if CMP0025 is set to NEW will detect apple-clang
    elseif((CONAN_COMPILER STREQUAL "gcc" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR
        (CONAN_COMPILER STREQUAL "apple-clang" AND NOT CROSS_BUILDING AND (NOT APPLE OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")) OR
        (CONAN_COMPILER STREQUAL "clang" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR
        (CONAN_COMPILER STREQUAL "sun-cc" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "SunPro") )
        message(FATAL_ERROR "Incorrect '${CONAN_COMPILER}', is not the one detected by CMake: '${CMAKE_CXX_COMPILER_ID}'")
    endif()


    if(NOT DEFINED CONAN_COMPILER_VERSION)
        conan_message(STATUS "WARN: CONAN_COMPILER_VERSION variable not set, please make sure yourself "
                             "that your compiler version matches your declared settings")
        return()
    endif()
    check_compiler_version()
endfunction()


macro(conan_set_flags build_type)
    set(CMAKE_CXX_FLAGS${build_type} "${CMAKE_CXX_FLAGS${build_type}} ${CONAN_CXX_FLAGS${build_type}}")
    set(CMAKE_C_FLAGS${build_type} "${CMAKE_C_FLAGS${build_type}} ${CONAN_C_FLAGS${build_type}}")
    set(CMAKE_SHARED_LINKER_FLAGS${build_type} "${CMAKE_SHARED_LINKER_FLAGS${build_type}} ${CONAN_SHARED_LINKER_FLAGS${build_type}}")
    set(CMAKE_EXE_LINKER_FLAGS${build_type} "${CMAKE_EXE_LINKER_FLAGS${build_type}} ${CONAN_EXE_LINKER_FLAGS${build_type}}")
endmacro()


macro(conan_global_flags)
    if(CONAN_SYSTEM_INCLUDES)
        include_directories(SYSTEM ${CONAN_INCLUDE_DIRS}
                                   "$<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_RELEASE}>"
                                   "$<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_RELWITHDEBINFO}>"
                                   "$<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_MINSIZEREL}>"
                                   "$<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_DEBUG}>")
    else()
        include_directories(${CONAN_INCLUDE_DIRS}
                            "$<$<CONFIG:Release>:${CONAN_INCLUDE_DIRS_RELEASE}>"
                            "$<$<CONFIG:RelWithDebInfo>:${CONAN_INCLUDE_DIRS_RELWITHDEBINFO}>"
                            "$<$<CONFIG:MinSizeRel>:${CONAN_INCLUDE_DIRS_MINSIZEREL}>"
                            "$<$<CONFIG:Debug>:${CONAN_INCLUDE_DIRS_DEBUG}>")
    endif()

    link_directories(${CONAN_LIB_DIRS})

    conan_find_libraries_abs_path("${CONAN_LIBS_DEBUG}" "${CONAN_LIB_DIRS_DEBUG}"
                                  CONAN_LIBS_DEBUG)
    conan_find_libraries_abs_path("${CONAN_LIBS_RELEASE}" "${CONAN_LIB_DIRS_RELEASE}"
                                  CONAN_LIBS_RELEASE)
    conan_find_libraries_abs_path("${CONAN_LIBS_RELWITHDEBINFO}" "${CONAN_LIB_DIRS_RELWITHDEBINFO}"
                                  CONAN_LIBS_RELWITHDEBINFO)
    conan_find_libraries_abs_path("${CONAN_LIBS_MINSIZEREL}" "${CONAN_LIB_DIRS_MINSIZEREL}"
                                  CONAN_LIBS_MINSIZEREL)

    add_compile_options(${CONAN_DEFINES}
                        "$<$<CONFIG:Debug>:${CONAN_DEFINES_DEBUG}>"
                        "$<$<CONFIG:Release>:${CONAN_DEFINES_RELEASE}>"
                        "$<$<CONFIG:RelWithDebInfo>:${CONAN_DEFINES_RELWITHDEBINFO}>"
                        "$<$<CONFIG:MinSizeRel>:${CONAN_DEFINES_MINSIZEREL}>")

    conan_set_flags("")
    conan_set_flags("_RELEASE")
    conan_set_flags("_DEBUG")

endmacro()


macro(conan_target_link_libraries target)
    if(CONAN_TARGETS)
        target_link_libraries(${target} ${CONAN_TARGETS})
    else()
        target_link_libraries(${target} ${CONAN_LIBS})
        foreach(_LIB ${CONAN_LIBS_RELEASE})
            target_link_libraries(${target} optimized ${_LIB})
        endforeach()
        foreach(_LIB ${CONAN_LIBS_DEBUG})
            target_link_libraries(${target} debug ${_LIB})
        endforeach()
    endif()
endmacro()


macro(conan_include_build_modules)
    if(CMAKE_BUILD_TYPE)
        if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_DEBUG} ${CONAN_BUILD_MODULES_PATHS})
        elseif(${CMAKE_BUILD_TYPE} MATCHES "Release")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_RELEASE} ${CONAN_BUILD_MODULES_PATHS})
        elseif(${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_RELWITHDEBINFO} ${CONAN_BUILD_MODULES_PATHS})
        elseif(${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel")
            set(CONAN_BUILD_MODULES_PATHS ${CONAN_BUILD_MODULES_PATHS_MINSIZEREL} ${CONAN_BUILD_MODULES_PATHS})
        endif()
    endif()

    foreach(_BUILD_MODULE_PATH ${CONAN_BUILD_MODULES_PATHS})
        include(${_BUILD_MODULE_PATH})
    endforeach()
endmacro()


### Definition of user declared vars (user_info) ###

set(CONAN_USER_FREETYPE_LIBTOOL_VERSION "24.0.18")