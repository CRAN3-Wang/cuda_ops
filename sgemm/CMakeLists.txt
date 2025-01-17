find_package(BLAS REQUIRED)

add_executable(sgemm src/sgemm.cu src/sgemm_v0_global_mem.cu src/utils.cpp)

target_include_directories(sgemm PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(sgemm PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(sgemm PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()