find_package(BLAS REQUIRED)

add_executable(reduce_v0_global_mem reduce_v0_global_mem.cu)
target_link_libraries(reduce_v0_global_mem PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v0_global_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v1_shared_mem reduce_v1_shared_mem.cu)
target_link_libraries(reduce_v1_shared_mem PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v1_shared_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v2_no_divergence_branch reduce_v2_no_divergence_branch.cu)
target_link_libraries(reduce_v2_no_divergence_branch PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v2_no_divergence_branch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v3_no_bank_conflict reduce_v3_no_bank_conflict.cu)
target_link_libraries(reduce_v3_no_bank_conflict PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v3_no_bank_conflict PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v4_add_during_load_A reduce_v4_add_during_load_A.cu)
target_link_libraries(reduce_v4_add_during_load_A PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v4_add_during_load_A PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v4_add_during_load_B reduce_v4_add_during_load_B.cu)
target_link_libraries(reduce_v4_add_during_load_B PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v4_add_during_load_B PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v5_unroll_last_warp reduce_v5_unroll_last_warp.cu)
target_link_libraries(reduce_v5_unroll_last_warp PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v5_unrool_last_warp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v6_completely_unroll reduce_v6_completely_unroll.cu)
target_link_libraries(reduce_v6_completely_unroll PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v6_completely_unroll PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v7_block_num reduce_v7_block_num.cu)
target_link_libraries(reduce_v7_block_num PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v7_block_num PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_v8_shuffle reduce_v8_shuffle.cu)
target_link_libraries(reduce_v8_shuffle PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v8_shuffle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(reduce_compare reduce_compare.cu)
target_link_libraries(reduce_compare PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_compare PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()