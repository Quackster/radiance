# A plugin like any other: built here inside radiance's tree, or on its own against an installed
# radiance (`cmake -S arch/embgemma2 -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`) -- the same
# rad_add_plugin either way (cmake/RadAddPlugin.cmake).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(embgemma2 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(embgemma2 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/embgemma2.cpp)
