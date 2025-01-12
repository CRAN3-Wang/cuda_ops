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
include ncu_profile/CMakeFiles/combined_access.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include ncu_profile/CMakeFiles/combined_access.dir/compiler_depend.make

# Include the progress variables for this target.
include ncu_profile/CMakeFiles/combined_access.dir/progress.make

# Include the compile flags for this target's objects.
include ncu_profile/CMakeFiles/combined_access.dir/flags.make

ncu_profile/CMakeFiles/combined_access.dir/combined_access.cu.o: ncu_profile/CMakeFiles/combined_access.dir/flags.make
ncu_profile/CMakeFiles/combined_access.dir/combined_access.cu.o: ../ncu_profile/combined_access.cu
ncu_profile/CMakeFiles/combined_access.dir/combined_access.cu.o: ncu_profile/CMakeFiles/combined_access.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/crane/dev/cuda_ops/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object ncu_profile/CMakeFiles/combined_access.dir/combined_access.cu.o"
	cd /home/crane/dev/cuda_ops/build/ncu_profile && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT ncu_profile/CMakeFiles/combined_access.dir/combined_access.cu.o -MF CMakeFiles/combined_access.dir/combined_access.cu.o.d -x cu -c /home/crane/dev/cuda_ops/ncu_profile/combined_access.cu -o CMakeFiles/combined_access.dir/combined_access.cu.o

ncu_profile/CMakeFiles/combined_access.dir/combined_access.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/combined_access.dir/combined_access.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

ncu_profile/CMakeFiles/combined_access.dir/combined_access.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/combined_access.dir/combined_access.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target combined_access
combined_access_OBJECTS = \
"CMakeFiles/combined_access.dir/combined_access.cu.o"

# External object files for target combined_access
combined_access_EXTERNAL_OBJECTS =

ncu_profile/combined_access: ncu_profile/CMakeFiles/combined_access.dir/combined_access.cu.o
ncu_profile/combined_access: ncu_profile/CMakeFiles/combined_access.dir/build.make
ncu_profile/combined_access: /usr/local/cuda/lib64/libcudart.so
ncu_profile/combined_access: /usr/local/cuda/lib64/libcublas.so
ncu_profile/combined_access: ncu_profile/CMakeFiles/combined_access.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/crane/dev/cuda_ops/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable combined_access"
	cd /home/crane/dev/cuda_ops/build/ncu_profile && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/combined_access.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ncu_profile/CMakeFiles/combined_access.dir/build: ncu_profile/combined_access
.PHONY : ncu_profile/CMakeFiles/combined_access.dir/build

ncu_profile/CMakeFiles/combined_access.dir/clean:
	cd /home/crane/dev/cuda_ops/build/ncu_profile && $(CMAKE_COMMAND) -P CMakeFiles/combined_access.dir/cmake_clean.cmake
.PHONY : ncu_profile/CMakeFiles/combined_access.dir/clean

ncu_profile/CMakeFiles/combined_access.dir/depend:
	cd /home/crane/dev/cuda_ops/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/crane/dev/cuda_ops /home/crane/dev/cuda_ops/ncu_profile /home/crane/dev/cuda_ops/build /home/crane/dev/cuda_ops/build/ncu_profile /home/crane/dev/cuda_ops/build/ncu_profile/CMakeFiles/combined_access.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ncu_profile/CMakeFiles/combined_access.dir/depend

