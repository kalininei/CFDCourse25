set (SRC
	cfd25.cpp

	mat/csrmat.cpp
	mat/lodmat.cpp
	mat/densemat.cpp
	mat/sparse_matrix_solver.cpp

	geom/simplex.cpp
	geom/searcher.cpp
	geom/jacobi.cpp

	grid/vtk.cpp
	grid/i_grid.cpp
	grid/grid1d.cpp
	grid/regular_grid2d.cpp
	grid/unstructured_grid2d.cpp

	numeric_integration/quadrature.cpp
	numeric_integration/segment_quadrature.cpp
	numeric_integration/square_quadrature.cpp
	numeric_integration/triangle_quadrature.cpp

	debug/printer.cpp
	debug/saver.cpp
	debug/tictoc.cpp
)

set (HEADERS
	cfd25.hpp
	cfd_common.hpp
	macros.hpp

	mat/i_mat.hpp
	mat/i_sparse_mat.hpp
	mat/csrmat.hpp
	mat/lodmat.hpp
	mat/sparse_matrix_solver.hpp
	mat/densemat.hpp

	geom/point.hpp
	geom/simplex.hpp
	geom/searcher.hpp
	geom/jacobi.hpp
	geom/i_point_function.hpp

	grid/vtk.hpp
	grid/i_grid.hpp
	grid/grid1d.hpp
	grid/regular_grid2d.hpp
	grid/unstructured_grid2d.hpp

	numeric_integration/quadrature.hpp
	numeric_integration/segment_quadrature.hpp
	numeric_integration/square_quadrature.hpp
	numeric_integration/triangle_quadrature.hpp

	debug/printer.hpp
	debug/saver.hpp
	debug/tictoc.hpp
)

add_library(${CFD_TARGET} STATIC ${SRC} ${HEADERS})
