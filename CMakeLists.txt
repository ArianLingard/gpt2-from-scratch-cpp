cmake_minimum_required(VERSION 3.18)
project(libtorch_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 1. 配置 LibTorch
set(CMAKE_PREFIX_PATH "D:/LibTorch")
find_package(Torch REQUIRED)

# 2. 编译可执行文件（源码已拆分到 src/ 下的多个文件）
add_executable(libtorch_demo
    src/main.cpp
    src/weights.cpp
    src/generate.cpp
    src/bpe.cpp
)

# 3. 链接 LibTorch，并包含 include 与 src 两个头文件目录
target_link_libraries(libtorch_demo "${TORCH_LIBRARIES}")
target_include_directories(libtorch_demo PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

if (MSVC)
    target_compile_options(libtorch_demo PRIVATE /W4 /utf-8 /wd4702)
endif()
