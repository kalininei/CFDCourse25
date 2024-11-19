set (SRC
	cfd25_test.cpp
	grid_test.cpp
	mat_test.cpp
	geom_test.cpp
	poisson_fdm_solve_test.cpp
	poisson_fvm_solve_test.cpp
	poisson_fem_solve_test.cpp
	fvm_assembly_test.cpp
	transport_solve_test.cpp
	transport_fvm_solve_test.cpp
	convdiff_fem_test.cpp
	
	utils/filesystem.cpp
	utils/vecmat.cpp
)

set (HEADERS
	cfd25_test.hpp

	utils/filesystem.hpp
	utils/vecmat.hpp
)

add_executable(cfd25_test ${SRC} ${HEADERS})
target_link_libraries(cfd25_test PRIVATE ${CATCH2_FRAMEWORK} ${CFD_TARGET})
target_include_directories(cfd25_test PRIVATE ${CMAKE_SOURCE_DIR}/src/test)
target_compile_definitions(cfd25_test PUBLIC TEST_DIRECTORY="${CMAKE_SOURCE_DIR}/test_data/")
