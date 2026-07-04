project(autograd_lib LANGUAGES CXX)

set(AUTOGRAD_SOURCES
  utils/constructor.cpp
  variable.cpp
  grad_mode.cpp
  utils/grad_utils.cpp
  functions/matrix_ops.cpp
  functions/unary_ops.cpp
  functions/reduct_ops.cpp
  functions/expand_ops.cpp
  functions/view_ops.cpp
)

add_library(autograd_core SHARED ${AUTOGRAD_SOURCES})

# Apply coverage flags if enabled
if(LMP_ENABLE_COVERAGE)
  target_compile_options(autograd_core PRIVATE --coverage -O0)
  target_link_options(autograd_core PRIVATE --coverage)
endif()

target_link_libraries(autograd_core
  PUBLIC tensor_core lamp3_compile_config
)

target_include_directories(autograd_core
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
