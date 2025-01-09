find_package(BLAS REQUIRED)
add_executable(reduce_v0_global_mem reduce_v0_global_mem.cu)
target_link_libraries(reduce_v0_global_mem PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v0_global_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()