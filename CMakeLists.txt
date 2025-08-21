cmake_minimum_required(VERSION 3.16)
project(urbanflow LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

# new relative path
set(CPP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/urbanflow/cpp/path_planner.cpp")
pybind11_add_module(path_planner ${CPP_SRC})

# name the final module urbanflow/_cpp/path_planner.*.so
set_target_properties(path_planner PROPERTIES
    OUTPUT_NAME     "path_planner"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/urbanflow/cpp"
)

target_compile_features(path_planner PRIVATE cxx_std_17)
