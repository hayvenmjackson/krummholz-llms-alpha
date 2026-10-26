cmake_minimum_required(VERSION 3.22.1)
project(krummholz_llm_android LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CI checks out this immutable b11669 revision; CMake never downloads anything.
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/llama.cpp"
    CACHE PATH "Pinned llama.cpp checkout")
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE LLAMA_SOURCE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LLAMA_REVISION_STATUS ERROR_QUIET)
if(NOT LLAMA_REVISION_STATUS EQUAL 0 OR
   NOT LLAMA_SOURCE_REVISION STREQUAL "620d2225cc436ab0b001e3200e590e3b286ef9ab")
    message(FATAL_ERROR "LLAMA_CPP_DIR must be the pinned llama.cpp b11669 checkout")
endif()

# Link only the CPU inference library into our JNI library. No CLI, server,
# download helper, plugin backend, OpenMP runtime or host-specific CPU flags.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_CPU_REPACK OFF CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
foreach(backend CUDA VULKAN METAL RPC BLAS OPENCL SYCL)
    set(GGML_${backend} OFF CACHE BOOL "" FORCE)
endforeach()
foreach(component COMMON TESTS TOOLS EXAMPLES SERVER APP UI)
    set(LLAMA_BUILD_${component} OFF CACHE BOOL "" FORCE)
endforeach()
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_ARM_ARCH "armv8-a" CACHE STRING "" FORCE)
elseif(ANDROID_ABI STREQUAL "x86_64")
    foreach(feature SSE42 AVX AVX2 AVX512 AVX_VNNI FMA F16C)
        set(GGML_${feature} OFF CACHE BOOL "" FORCE)
    endforeach()
else()
    message(FATAL_ERROR "Only arm64-v8a and x86_64 are supported")
endif()

add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama")
add_library(krummholz_llm SHARED native_engine.cpp)
target_link_libraries(krummholz_llm PRIVATE llama)
target_compile_options(krummholz_llm PRIVATE -Wall -Wextra -Werror)
target_link_options(krummholz_llm PRIVATE
    "-Wl,-z,max-page-size=16384" "-Wl,-z,common-page-size=16384")
