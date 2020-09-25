cmake_minimum_required(VERSION 3.10)

project("Vulkan Tutorial")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

set(SOURCES main.cpp)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/shaders" "${CMAKE_BINARY_DIR}/shaders"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/textures" "${CMAKE_BINARY_DIR}/textures"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/models" "${CMAKE_BINARY_DIR}/models")

add_executable(vulkan-tutorial ${SOURCES})
target_link_libraries(vulkan-tutorial glfw Vulkan::Vulkan)
