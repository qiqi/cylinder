cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(Cylinder)

execute_process(COMMAND python3 -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RESULT_VARIABLE NUMPY_NOT_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)

set(CUDA_MODE 1)
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH sm_60)
endif(NOT DEFINED CUDA_ARCH)

set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS} -I${NUMPY_INCLUDE_DIR}")
set(CUDA_NVCC_FLAGS -ccbin g++;${CUDA_NVCC_FLAGS};-arch=${CUDA_ARCH}; --std=c++14;-Wno-deprecated-gpu-targets; -lineinfo; -use_fast_math; -DFLOW360_GPU)

SET(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

#SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
#LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
#IF("${isSystemDir}" STREQUAL "-1")
#   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#ENDIF("${isSystemDir}" STREQUAL "-1")

# Module-specific defs
set(SOURCES poisson.cpp)

set_source_files_properties(
  ${SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

cuda_add_library(cylinder SHARED ${SOURCES} OPTIONS)

target_link_libraries(cylinder ${CUDA_LIBRARIES})

install(TARGETS cylinder DESTINATION lib)

ENABLE_TESTING()
add_subdirectory(tests)
