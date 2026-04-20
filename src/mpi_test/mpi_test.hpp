#ifndef CFD24_MPI_TEST_HPP
#define CFD24_MPI_TEST_HPP

#include <iostream>
#include <fstream>
#include "cfd/cfd25.hpp"
#include "cfd/macros.hpp"
#include "mpi.h"

#define CHECK(cond)\
	if (!(cond)){ \
		std::cout << "==== TEST FAILED" << std::endl; \
		std::cout << __PRETTY_FUNCTION__ << std::endl; \
		std::cout << "At" << std::endl; \
		std::cout << __FILE__ << ": " << __LINE__ << std::endl; \
		MPI_Abort(MPI_COMM_WORLD, 1); \
	}

std::string test_directory_file(std::string path);
void schwarz_ddm_poisson_test();

#endif
