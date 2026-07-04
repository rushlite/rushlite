project(tests LANGUAGES CXX)

# FETCH CONTENT ---------------------------------
include(FetchContent)
set(FETCHCONTENT_TRY_FIND_PACKAGE TRUE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS CONFIG 
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# EXECUTABLES ---------------------------------
enable_testing()

add_executable(autograd_tests autograd_tests.cpp)
target_link_libraries(autograd_tests
  PRIVATE autograd_core GTest::gtest_main GTest::gmock
)

add_executable(tensor_tests tensor_tests.cpp)
target_link_libraries(tensor_tests
  PRIVATE tensor_core GTest::gtest_main GTest::gmock
)

add_executable(playground playground.cpp)
target_link_libraries(playground
  PRIVATE autograd_core tensor_core nets_core
)

if(LMP_ENABLE_COVERAGE)
  target_compile_options(tensor_tests PRIVATE --coverage -O0 -g)
  target_link_options(tensor_tests PRIVATE --coverage)
  target_compile_options(autograd_tests PRIVATE --coverage -O0 -g)
  target_link_options(autograd_tests PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(tensor_tests PROPERTIES LABELS "library_tests")
gtest_discover_tests(autograd_tests PROPERTIES LABELS "library_tests")
