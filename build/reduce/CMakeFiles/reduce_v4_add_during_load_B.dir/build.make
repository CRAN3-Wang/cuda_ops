# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/crane/dev/cuda_ops

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/crane/dev/cuda_ops/build

# Include any dependencies generated for this target.
include reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/compiler_depend.make

# Include the progress variables for this target.
include reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/progress.make

# Include the compile flags for this target's objects.
include reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/flags.make

reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.o: reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/flags.make
reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.o: ../reduce/reduce_v4_add_during_load_B.cu
reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.o: reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/crane/dev/cuda_ops/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.o"
	cd /home/crane/dev/cuda_ops/build/reduce && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.o -MF CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.o.d -x cu -c /home/crane/dev/cuda_ops/reduce/reduce_v4_add_during_load_B.cu -o CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.o

reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target reduce_v4_add_during_load_B
reduce_v4_add_during_load_B_OBJECTS = \
"CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.o"

# External object files for target reduce_v4_add_during_load_B
reduce_v4_add_during_load_B_EXTERNAL_OBJECTS =

reduce/reduce_v4_add_during_load_B: reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/reduce_v4_add_during_load_B.cu.o
reduce/reduce_v4_add_during_load_B: reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/build.make
reduce/reduce_v4_add_during_load_B: /usr/local/cuda/lib64/libcudart.so
reduce/reduce_v4_add_during_load_B: /usr/local/cuda/lib64/libcublas.so
reduce/reduce_v4_add_during_load_B: reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/crane/dev/cuda_ops/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable reduce_v4_add_during_load_B"
	cd /home/crane/dev/cuda_ops/build/reduce && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduce_v4_add_during_load_B.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/build: reduce/reduce_v4_add_during_load_B
.PHONY : reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/build

reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/clean:
	cd /home/crane/dev/cuda_ops/build/reduce && $(CMAKE_COMMAND) -P CMakeFiles/reduce_v4_add_during_load_B.dir/cmake_clean.cmake
.PHONY : reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/clean

reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/depend:
	cd /home/crane/dev/cuda_ops/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/crane/dev/cuda_ops /home/crane/dev/cuda_ops/reduce /home/crane/dev/cuda_ops/build /home/crane/dev/cuda_ops/build/reduce /home/crane/dev/cuda_ops/build/reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : reduce/CMakeFiles/reduce_v4_add_during_load_B.dir/depend

