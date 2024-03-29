#include "cfd25_test.hpp"
#include "cfd/grid/grid1d.hpp"
#include "cfd/grid/unstructured_grid2d.hpp"
#include "cfd/mat/csrmat.hpp"
#include "cfd/mat/sparse_matrix_solver.hpp"
#include "cfd/grid/vtk.hpp"
#include "cfd/mat/lodmat.hpp"
#include "utils/filesystem.hpp"

using namespace cfd;

///////////////////////////////////////////////////////////////////////////////
// Poisson 1D solver
///////////////////////////////////////////////////////////////////////////////

namespace {

class TestPoissonFvm1Worker{
public:
	// u(x) = sin(10*x^2)
	static double exact_solution(Point p){
		double x = p.x();
		return sin(10*x*x);
	}
	
	// -d^2 u(x)/d x^2
	static double exact_rhs(Point p){
		double x = p.x();
		return 400*x*x*sin(10*x*x) - 20*cos(10*x*x);
	}

	TestPoissonFvm1Worker(size_t n_cells): _grid(0, 1, n_cells){
		// assemble face lists
		for (size_t iface=0; iface<_grid.n_faces(); ++iface){
			size_t icell_negative = _grid.tab_face_cell(iface)[0];
			size_t icell_positive = _grid.tab_face_cell(iface)[1];
			if (icell_positive != INVALID_INDEX && icell_negative != INVALID_INDEX){
				// internal faces list
				_internal_faces.push_back(iface);
			} else {
				// dirichlet faces list
				DirichletFace dir_face;
				dir_face.iface = iface;
				dir_face.value = exact_solution(_grid.face_center(iface));
				if (icell_positive == INVALID_INDEX){
					dir_face.icell = icell_negative;
					dir_face.outer_normal = _grid.face_normal(iface);
				} else {
					dir_face.icell = icell_positive;
					dir_face.outer_normal = -_grid.face_normal(iface);
				}
				_dirichlet_faces.push_back(dir_face);
			}
		}
	}

	// returns norm2(u - u_exact)
	double solve(){
		// 1. build SLAE
		CsrMatrix mat = approximate_lhs();
		std::vector<double> rhs = approximate_rhs();

		// 2. solve SLAE
		AmgcMatrixSolver solver;
		solver.set_matrix(mat);
		solver.solve(rhs, _u);

		// 3. compute norm2
		return compute_norm2();
	}

	// saves numerical and exact solution into the vtk format
	void save_vtk(const std::string& filename){
		// save grid
		_grid.save_vtk(filename);
		
		// save numerical solution
		VtkUtils::add_cell_data(_u, "numerical", filename);

		// save exact solution
		std::vector<double> exact(_grid.n_cells());
		for (size_t i=0; i<_grid.n_cells(); ++i){
			exact[i] = exact_solution(_grid.cell_center(i));
		}
		VtkUtils::add_cell_data(exact, "exact", filename);
	}

private:
	const Grid1D _grid;
	std::vector<size_t> _internal_faces;
	struct DirichletFace{
		size_t iface;
		size_t icell;
		double value;
		Vector outer_normal;
	};
	std::vector<DirichletFace> _dirichlet_faces;
	std::vector<double> _u;

	CsrMatrix approximate_lhs() const{
		LodMatrix mat(_grid.n_cells());
		// internal faces
		for (size_t iface: _internal_faces){
			size_t negative_side_cell = _grid.tab_face_cell(iface)[0];
			size_t positive_side_cell = _grid.tab_face_cell(iface)[1];
			Point ci = _grid.cell_center(negative_side_cell);
			Point cj = _grid.cell_center(positive_side_cell);
			double h = vector_abs(cj - ci);
			double coef = _grid.face_area(iface) / h;

			mat.add_value(negative_side_cell, negative_side_cell, coef);
			mat.add_value(positive_side_cell, positive_side_cell, coef);
			mat.add_value(negative_side_cell, positive_side_cell, -coef);
			mat.add_value(positive_side_cell, negative_side_cell, -coef);
		}
		// dirichlet faces
		for (const DirichletFace& dir_face: _dirichlet_faces){
			size_t icell = dir_face.icell;
			size_t iface = dir_face.iface;
			Point gs = _grid.face_center(iface);
			Point ci = _grid.cell_center(icell);
			double h = vector_abs(gs - ci);
			double coef = _grid.face_area(iface) / h;
			mat.add_value(icell, icell, coef);
		}
		return mat.to_csr();
	}

	std::vector<double> approximate_rhs() const{
		std::vector<double> rhs(_grid.n_cells(), 0.0);
		// internal
		for (size_t icell=0; icell < _grid.n_cells(); ++icell){
			double value = exact_rhs(_grid.cell_center(icell));
			double volume = _grid.cell_volume(icell);
			rhs[icell] = value * volume;
		}
		// dirichlet faces
		for (const DirichletFace& dir_face: _dirichlet_faces){
			size_t icell = dir_face.icell;
			size_t iface = dir_face.iface;
			Point gs = _grid.face_center(iface);
			Point ci = _grid.cell_center(icell);
			double h = vector_abs(gs-ci);
			double coef = _grid.face_area(iface) / h;
			rhs[icell] += dir_face.value * coef;
		}
		return rhs;
	}

	double compute_norm2() const{
		double norm2 = 0;
		double full_area = 0;
		for (size_t icell=0; icell<_grid.n_cells(); ++icell){
			double diff = _u[icell] - exact_solution(_grid.cell_center(icell));
			norm2 += _grid.cell_volume(icell) * diff * diff;
			full_area += _grid.cell_volume(icell);
		}
		return std::sqrt(norm2/full_area);
	}
};
}

TEST_CASE("Poisson 1D solver, Finite Volume Method", "[poisson1-fvm]"){
	std::cout << std::endl << "--- [poisson1-fvm] --- " << std::endl;

	// precalculated norm2 results for some n_cells values
	// used for CHECK procedures
	std::map<size_t, double> norm2_for_compare{
		{10, 0.106539},
		{100, 0.00101714},
		{1000, 1.01641e-05},
	};

	// loop over n_cells value
	for (size_t n_cells: {10, 20, 50, 100, 200, 500, 1000}){
		// build test solver
		TestPoissonFvm1Worker worker(n_cells);

		// solve and find norm2
		double n2 = worker.solve();

		// save into poisson1_ncells={n_cells}.vtk
		worker.save_vtk("poisson1_fvm_n=" + std::to_string(n_cells) + ".vtk");

		// print (N_CELLS, NORM2) table entry
		std::cout << n_cells << " " << n2 << std::endl;

		// CHECK if result for this n_cells
		// presents in the norm2_for_compare dictionary
		auto found = norm2_for_compare.find(n_cells);
		if (found != norm2_for_compare.end()){
			CHECK(n2 == Approx(found->second).margin(1e-6));
		}
	}
}
