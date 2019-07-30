cmake_minimum_required(VERSION 3.10)

include(SetUpConfigurations.cmake)

project("Vulkan Test")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-extra-arg=-std=c++17;-checks=-clang-analyzer*,bugprone*,misc*,modernize*,performance*,readability*")

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

set(SOURCES main.cpp)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/shaders" "${CMAKE_BINARY_DIR}/shaders"
                COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/textures" "${CMAKE_BINARY_DIR}/textures"
                COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/models" "${CMAKE_BINARY_DIR}/models")

add_executable(vktest ${SOURCES})
target_link_libraries(vktest glfw Vulkan::Vulkan)
