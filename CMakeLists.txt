cmake_minimum_required(VERSION 3.10)

include(SetUpConfigurations.cmake)

project("Vulkan Test")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

set(SOURCES main.cpp)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/shaders" "${CMAKE_BINARY_DIR}/shaders"
                COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/textures" "${CMAKE_BINARY_DIR}/textures"
                COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_SOURCE_DIR}/models" "${CMAKE_BINARY_DIR}/models")

add_executable(vktest ${SOURCES})
target_link_libraries(vktest glfw Vulkan::Vulkan)

add_custom_command(TARGET vktest POST_BUILD
                   COMMAND "cppcheck"
                           "--project=compile_commands.json"
                           "--std=c++17"
                           "--template=gcc"
                           "--enable=all"
                           "--inline-suppr"
                           "--suppress=knownConditionTrueFalse"
                           "--suppress=*:*/stb_image.h"
                           "--suppress=missingIncludeSystem"
                           "--suppress=unmatchedSuppression:*")

# add_custom_command(TARGET vktest POST_BUILD
#                    COMMAND "run-clang-tidy" "-quiet" "-extra-arg=-std=c++17"
#                            "-checks=-clang-analyzer*,bugprone*,misc*,modernize*,performance*,readability*")
