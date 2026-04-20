cmake_minimum_required(VERSION 3.0)

Project(cfdcourse25)

set(CMAKE_CXX_STANDARD 17)

# Catch framework
include(FetchContent)
FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.10
)
FetchContent_GetProperties(catch)
if(NOT catch_POPULATED)
  FetchContent_Populate(catch)
  #add_subdirectory(${catch_SOURCE_DIR} ${catch_BINARY_DIR})
endif()
#set(CATCH2_FRAMEWORK "Catch2::Catch2")

# Boost search paths
set(BOOST_ROOT ${CMAKE_SOURCE_DIR}/../)
find_package(Boost REQUIRED)

# ------ amgcl
FetchContent_Declare(
	amgcl
	GIT_REPOSITORY https://github.com/ddemidov/amgcl
	GIT_TAG 1.4.2
)
FetchContent_GetProperties(amgcl)
if(NOT amgcl_POPULATED)
	FetchContent_Populate(amgcl)
endif()
include_directories(${amgcl_SOURCE_DIR})

# -------- umfpack
SET(UMFPACK_SEARCH_PATH /usr/lib64/ /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
find_path(UMFPACK_INCLUDE_DIR
    NAMES umfpack.h
    PATH_SUFFIXES suitesparse
    PATHS /usr/include /usr/local/include
)
find_library(UMFPACK_LIBRARY
    NAMES umfpack
    PATHS ${UMFPACK_SEARCH_PATH}
)
find_library(AMD_LIBRARY
    NAMES amd
    PATHS ${UMFPACK_SEARCH_PATH}
)
find_library(SUITESPARSE_CONFIG_LIBRARY
    NAMES suitesparseconfig
    PATHS ${UMFPACK_SEARCH_PATH}
)
set(UMFPACK_LIBRARIES
    ${UMFPACK_LIBRARY}
    ${AMD_LIBRARY}
    ${SUITESPARSE_CONFIG_LIBRARY}
)

# ------- ghc
FetchContent_Declare(
	ghc
	GIT_REPOSITORY https://github.com/gulrak/filesystem
	GIT_TAG v1.5.14
)
FetchContent_GetProperties(ghc)
if(NOT ghc_POPULATED)
	FetchContent_Populate(ghc)
endif()
include_directories(${ghc_SOURCE_DIR}/include)

# -------- mpi
option(MPI_TESTS "Enable MPI tests" OFF)
if (MPI_TESTS)
	find_package(MPI REQUIRED)
endif()

# build all binaries to build/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${Boost_INCLUDE_DIRS})

add_subdirectory(src)
