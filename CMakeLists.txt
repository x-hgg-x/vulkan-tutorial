cmake_minimum_required(VERSION 3.10)

project("Vulkan Tutorial")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

set(SOURCES main.cpp)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/assets" "${CMAKE_BINARY_DIR}/assets")

add_executable(vulkan-tutorial ${SOURCES})
target_link_libraries(vulkan-tutorial glfw Vulkan::Vulkan)
