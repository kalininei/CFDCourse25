set(CFD_TARGET cfd25)

add_subdirectory(cfd)
add_subdirectory(test)
if (MPI_TESTS)
	add_subdirectory(mpi_test)
endif()
