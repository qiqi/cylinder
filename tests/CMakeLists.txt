find_package(PythonLibs 3 REQUIRED)
message("Python libs: ${PYTHON_LIBRARIES}")
include_directories(".." "${CMAKE_CURRENT_SOURCE_DIR}/../"
                    "${PYTHON_INCLUDE_DIRS}" "${CUDA_INCLUDE_DIRS}")

IF(DEFINED EXTRA_INCLUDES)
  include_directories("${EXTRA_INCLUDES}")
ENDIF(DEFINED EXTRA_INCLUDES)

set_source_files_properties(
    test_poisson.cpp
    PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

cuda_add_executable(test_poisson test_poisson.cpp)
target_link_libraries(test_poisson cylinder)

#configure_file(testSubsonicInflow.py testSubsonicInflow.py COPYONLY)

#add_test(testRoeFlux python3 testRoeFlux.py)
