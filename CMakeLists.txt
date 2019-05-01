cmake_minimum_required(VERSION 3.10)

include(SetUpConfigurations.cmake)

project("Vulkan Test")

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

set(SOURCES main.cpp)

file(GLOB SHADERS "${CMAKE_SOURCE_DIR}/shaders/*.spv")
file(COPY ${SHADERS} DESTINATION "${CMAKE_BINARY_DIR}/shaders")

add_executable(vktest ${SOURCES})
target_link_libraries(vktest glfw Vulkan::Vulkan)
