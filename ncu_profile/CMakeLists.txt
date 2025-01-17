add_executable(combined_access combined_access.cu)
target_link_libraries(combined_access PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(combined_access PRIVATE -lineinfo)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(combined_access PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(transpose transpose.cu)
target_link_libraries(transpose PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_compile_options(transpose PRIVATE -lineinfo)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()