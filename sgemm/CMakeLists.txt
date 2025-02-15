find_package(BLAS REQUIRED)

add_executable(sgemm src/sgemm.cu 
                src/sgemm_v0_global_mem.cu 
                src/sgemm_v1_shared_mem.cu 
                src/sgemm_v2_increase_workload_of_threads.cu 
                src/sgemm_v3_float4.cu 
                src/sgemm_v4_reg.cu 
                src/sgemm_v5_reg_float4.cu
                src/sgemm_v6_transpose_A_smem.cu
                src/sgemm_v7_double_buffer.cu
                src/sgemm_v8_double_reg.cu
                src/sgemm_comp.cu
                src/sgemm_utils.cpp)

target_include_directories(sgemm PRIVATE ${CMAKE_SOURCE_DIR}/sgemm/include)

target_link_libraries(sgemm PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(sgemm PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()