cmake_minimum_required(VERSION 3.27)
project(lamp3 LANGUAGES CXX)

option(LMP_ENABLE_CUDA "Compile with CUDA support" OFF)
option(LMP_ENABLE_COVERAGE "Enable code coverage" OFF)
option(LMP_ENABLE_TEST "Enable testing" OFF)
option(LMP_ENABLE_BENCH "Enable benchmarks" OFF)

if(LMP_ENABLE_CUDA)
  add_definitions(-DLMP_ENABLE_CUDA)
endif()

# CORE CONFIG ---------------------------------
set(PYTHON_VENV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")

if(EXISTS ${PYTHON_VENV_PATH})
  set(Python3_EXECUTABLE ${PYTHON_VENV_PATH} CACHE FILEPATH "Path to Python3 executable from Poetry venv" FORCE)
  message(STATUS "Attempting to use Python executable from Poetry venv: ${Python3_EXECUTABLE}")
else()
  message(WARNING "Poetry virtual environment Python not found at ${PYTHON_VENV_PATH}. CMake will attempt to find system Python. This may lead to version mismatches.")
endif()

# Expected python version is >=3.11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "CMake is using Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 version: ${Python3_VERSION_STRING}")
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

# Setting C++ and CUDA versions
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (LMP_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  set(CMAKE_CUDA_ARCHITECTURES native)
  find_package(CUDAToolkit REQUIRED)              
endif()

find_package(OpenMP)

# Enabled for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CORE TARGETS ---------------------------------
set(DEFAULT_DTYPE "Float32")
if(LMP_ENABLE_CUDA)
  set(DEFAULT_DEVICE "CUDA")
else()
  set(DEFAULT_DEVICE "CPU")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include/lamp3/common/config.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/csrc/include/lamp3/common/config.hpp"
)

add_library(lamp3_compile_config INTERFACE)
target_compile_options(lamp3_compile_config
  INTERFACE
    $<$<CONFIG:Release>:-march=native;-ffast-math> # also compiles with -O3 -DNDEBUG
    $<$<CONFIG:Debug>:-march=native;-ffast-math;-DLMP_DEBUG> # -g
)
target_include_directories(lamp3_compile_config
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/csrc/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/csrc/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_subdirectory(csrc/src/tensor)
add_subdirectory(csrc/src/inductor)
add_subdirectory(csrc/src/autograd)
add_subdirectory(csrc/src/nets)
add_library(lamp3 INTERFACE)
target_link_libraries(lamp3 INTERFACE autograd_core tensor_core inductor_core nets_core)

add_subdirectory(docs) # TODO: this includes some examples that we don't technically need to build
add_subdirectory(csrc/bindings)
if(LMP_ENABLE_TEST)
  add_subdirectory(csrc/tests)
endif()
if(LMP_ENABLE_BENCH)
  add_subdirectory(benchmarks)
endif()

# INSTALLATION ---------------------------------
install(
  TARGETS
    lamp3
    autograd_core
    tensor_core
    inductor_core
    nets_core
    lamp3_compile_config
  EXPORT lamp3Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/rushlite/lib
  RUNTIME DESTINATION bin
)

install(
  EXPORT lamp3Targets
  FILE   lamp3Targets.cmake
  DESTINATION lib/cmake/lamp3
)

install(
  DIRECTORY csrc/include/
  DESTINATION include
)

# STATIC ANALYSIS ---------------------------------
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(CLANG_TIDY_EXE)
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

# FORMATTING ---------------------------------
file(GLOB_RECURSE ALL_CPP_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/src/tensor/*.[ch]pp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/src/autograd/*.[ch]pp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/src/bindings/*.[ch]pp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/tests/*.[ch]pp"
  "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.[ch]pp" 
)

file(GLOB_RECURSE ALL_CUDA_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/src/tensor/*.[ch]u"
)
set(ALL_SOURCE ${ALL_CPP_SOURCE} ${ALL_CUDA_SOURCE})

find_program(CLANG_FORMAT_EXE NAMES clang-format)
if(CLANG_FORMAT_EXE)

  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE}
            -i
            ${ALL_SOURCE}
    COMMENT "Apply clang-format to all sources"
  )
endif()


# LINTING ---------------------------------
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(CLANG_TIDY_EXE)
  set(HEADER_FILTER_REGEX "${CMAKE_CURRENT_SOURCE_DIR}/(csrc|benchmarks)/.*")

  if (LMP_ENABLE_CUDA)
    add_custom_target(lint
      COMMAND ${CLANG_TIDY_EXE}
              -p build
              ${ALL_SOURCE} -header-filter="${HEADER_FILTER_REGEX}"
      COMMENT "Apply clang-tidy to all sources"
    )

    add_custom_target(lint-fix
      COMMAND ${CLANG_TIDY_EXE}
              -p build -fix-errors
              ${ALL_SOURCE} -header-filter="${HEADER_FILTER_REGEX}"
      COMMENT "Apply clang-tidy to all sources"
    )
  else()
    add_custom_target(lint
      COMMAND ${CLANG_TIDY_EXE}
              -p build
              ${ALL_CPP_SOURCE} -header-filter="${HEADER_FILTER_REGEX}"
      COMMENT "No CUDA detected, applying clang-tidy to all C++ sources"
    )

    add_custom_target(lint-fix
      COMMAND ${CLANG_TIDY_EXE}
              -p build -fix-errors
              ${ALL_CPP_SOURCE} -header-filter="${HEADER_FILTER_REGEX}"
      COMMENT "No CUDA detected, applying clang-tidy to all C++ sources"
    )
  endif()

endif()


# COVERAGE ---------------------------------
if(LMP_ENABLE_COVERAGE)
  include(CTest)
  set(CMAKE_CXX_FLAGS_DEBUG "-fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "--coverage")
  set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "--coverage")
  message(STATUS "Code coverage enabled")
  
  add_custom_target(coverage_clean
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/coverage.info ${CMAKE_BINARY_DIR}/python_cpp_coverage.info ${CMAKE_BINARY_DIR}/cpp_filtered.info ${CMAKE_BINARY_DIR}/python_filtered.info ${CMAKE_BINARY_DIR}/combined_coverage.info
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/coverage || true
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Cleaning up previous coverage artifacts"
  )

  add_custom_target(coverage_cpp
    COMMAND ctest --test-dir ${CMAKE_BINARY_DIR} -L library_tests -T Test -T Coverage
    COMMAND lcov --directory ${CMAKE_BINARY_DIR} --capture --output-file ${CMAKE_BINARY_DIR}/coverage.info --ignore-errors mismatch,gcov,negative,source --rc geninfo_unexecuted_blocks=1
    COMMAND lcov --extract ${CMAKE_BINARY_DIR}/coverage.info '${CMAKE_SOURCE_DIR}/csrc/src/*' '${CMAKE_SOURCE_DIR}/csrc/include/*' -o ${CMAKE_BINARY_DIR}/cpp_filtered.info
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running C++ tests and generating coverage data"
    DEPENDS coverage_clean
  )

  add_custom_target(coverage_python
    COMMAND poetry run pytest tests/
    COMMAND lcov --directory ${CMAKE_BINARY_DIR} --capture --output-file ${CMAKE_BINARY_DIR}/python_cpp_coverage.info --ignore-errors mismatch,gcov,negative,source --rc geninfo_unexecuted_blocks=1
    COMMAND lcov --extract ${CMAKE_BINARY_DIR}/python_cpp_coverage.info '${CMAKE_SOURCE_DIR}/csrc/src/*' '${CMAKE_SOURCE_DIR}/csrc/include/*' -o ${CMAKE_BINARY_DIR}/python_filtered.info
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running Python tests and capturing C++ coverage data"
    DEPENDS coverage_clean
  )

  add_custom_target(coverage
    COMMAND lcov --add-tracefile ${CMAKE_BINARY_DIR}/cpp_filtered.info --add-tracefile ${CMAKE_BINARY_DIR}/python_filtered.info --output-file ${CMAKE_BINARY_DIR}/combined_coverage.info || lcov --add-tracefile ${CMAKE_BINARY_DIR}/cpp_filtered.info --output-file ${CMAKE_BINARY_DIR}/combined_coverage.info
    COMMAND genhtml ${CMAKE_BINARY_DIR}/combined_coverage.info --output-directory ${CMAKE_BINARY_DIR}/coverage
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Combining coverage reports and generating HTML output"
    DEPENDS coverage_cpp coverage_python
  )
endif()