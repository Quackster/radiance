# A plugin like any other: built here inside radiance's tree, or on its own against an installed
# radiance (`cmake -S arch/qwen35_mxfp4 -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`) -- the
# same rad_add_plugin either way (cmake/RadAddPlugin.cmake).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(qwen35_mxfp4 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(qwen35_mxfp4 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/qwen35_mxfp4.cpp)
