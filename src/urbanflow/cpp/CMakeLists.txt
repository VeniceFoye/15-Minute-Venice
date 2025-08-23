cmake_minimum_required(VERSION 3.21)
project(urbanflow_cpp LANGUAGES CXX)

# Find Python & NumPy
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
# Find pybind11 (scikit-build-core will provide it from pyproject deps)
find_package(pybind11 CONFIG REQUIRED)

# Build extension: module name = urbanflow.cpp.path_planner
pybind11_add_module(path_planner MODULE path_planner.cpp)
target_compile_features(path_planner PRIVATE cxx_std_17)

# Include dirs and link Python/NumPy if needed
target_include_directories(path_planner PRIVATE ${Python_NumPy_INCLUDE_DIRS})

# Ensure the module installs to the Python package path urbanflow/cpp
set_target_properties(path_planner PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  # for in-tree dev builds
)
