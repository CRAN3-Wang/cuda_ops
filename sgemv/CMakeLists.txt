find_package(BLAS REQUIRED)

add_executable(sgemv src/sgemv.cu
                src/sgemv_v0_32.cu
                src/sgemv_v1_float4.cu
                src/sgemv_v2_16.cu
                src/sgemv_utils.cpp)

target_include_directories(sgemv PRIVATE ${CMAKE_SOURCE_DIR}/sgemv/include)

target_link_libraries(sgemv PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgemv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()