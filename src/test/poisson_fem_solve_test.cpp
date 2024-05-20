#include "cfd25_test.hpp"
#include "test/utils/filesystem.hpp"
#include "cfd/grid/grid1d.hpp"
#include "cfd/mat/csrmat.hpp"
#include "cfd/mat/sparse_matrix_solver.hpp"
#include "cfd/fem/fem_assembler.hpp"
#include "cfd/fem/elem1d/segment_linear.hpp"
#include "cfd/grid/vtk.hpp"
#include "cfd/debug/printer.hpp"
#include "cfd/debug/saver.hpp"

using namespace cfd;

namespace{

///////////////////////////////////////////////////////////////////////////////
// ITestPoissonFemWorker
///////////////////////////////////////////////////////////////////////////////
struct ITestPoissonFemWorker{
	virtual double exact_solution(Point p) const = 0;
	virtual double exact_rhs(Point p) const = 0;
	virtual std::vector<size_t> dirichlet_bases() const{
		return _grid.boundary_points();
	}

	ITestPoissonFemWorker(const IGrid& grid, const FemAssembler& fem);
	virtual double solve();
	void save_vtk(const std::string& filename) const;
protected:
	const IGrid& _grid;
	FemAssembler _fem;
	std::vector<double> _u;

	CsrMatrix approximate_lhs() const;
	std::vector<double> approximate_rhs() const;
	double compute_norm2() const;
};

ITestPoissonFemWorker::ITestPoissonFemWorker(const IGrid& grid, const FemAssembler& fem): _grid(grid), _fem(fem){
}

double ITestPoissonFemWorker::solve(){
	// 1. build SLAE
	CsrMatrix mat = approximate_lhs();
	std::vector<double> rhs = approximate_rhs();
	// 2. Dirichlet bc
	for (size_t ibas: dirichlet_bases()){
		mat.set_unit_row(ibas);
		Point p = _fem.reference_point(ibas);
		rhs[ibas] = exact_solution(p);
	}
	// 3. solve SLAE
	AmgcMatrixSolver solver({ {"precond.relax.type", "gauss_seidel"} });
	solver.set_matrix(mat);
	solver.solve(rhs, _u);
	// 4. compute norm2
	return compute_norm2();
}

void ITestPoissonFemWorker::save_vtk(const std::string& filename) const{
	// save grid
	_grid.save_vtk(filename);
	// save numerical solution
	VtkUtils::add_point_data(_u, "numerical", filename, _grid.n_points());
	// save exact solution
	std::vector<double> exact(_grid.n_points());
	for (size_t i=0; i<_grid.n_points(); ++i){
		exact[i] = exact_solution(_grid.point(i));
	}
	VtkUtils::add_point_data(exact, "exact", filename, _grid.n_points());
}

CsrMatrix ITestPoissonFemWorker::approximate_lhs() const{
	CsrMatrix ret(_fem.stencil());
	for (size_t ielem=0; ielem < _fem.n_elements(); ++ielem){
		const FemElement& elem = _fem.element(ielem);
		std::vector<double> local_stiff = elem.integrals->stiff_matrix();
		_fem.add_to_global_matrix(ielem, local_stiff, ret.vals());
	}
	return ret;
}

std::vector<double> ITestPoissonFemWorker::approximate_rhs() const{
	// mass matrix
	CsrMatrix mass(_fem.stencil());
	for (size_t ielem=0; ielem < _fem.n_elements(); ++ielem){
		const FemElement& elem = _fem.element(ielem);
		std::vector<double> local_mass = elem.integrals->mass_matrix();
		_fem.add_to_global_matrix(ielem, local_mass, mass.vals());
	}
	// rhs = Mass * f
	std::vector<double> fvec(_fem.n_bases());
	for (size_t ibas=0; ibas < _fem.n_bases(); ++ibas){
		Point p = _fem.reference_point(ibas);
		fvec[ibas] = exact_rhs(p);
	}
	return mass.mult_vec(fvec);
}

double ITestPoissonFemWorker::compute_norm2() const{
	std::vector<double> force_vec(_fem.n_bases(), 0);
	for (size_t ielem=0; ielem < _fem.n_elements(); ++ielem){
		const FemElement& elem = _fem.element(ielem);
		std::vector<double> v = elem.integrals->load_vector();
		_fem.add_to_global_vector(ielem, v, force_vec);
	}
	double integral = 0;
	double full_area = 0;
	for (size_t ibas=0; ibas<_fem.n_bases(); ++ibas){
		Point p = _fem.reference_point(ibas);
		double diff = _u[ibas] - exact_solution(p);
		integral += force_vec[ibas] * (diff * diff);
		full_area += force_vec[ibas];
	}
	return std::sqrt(integral/full_area);
}

////////////////////////////////////////////////////////////////////////////////
// ITestPoisson1FemWorker
////////////////////////////////////////////////////////////////////////////////
struct ITestPoisson1FemWorker: public ITestPoissonFemWorker{
	double exact_solution(Point p) const override{
		double x = p.x();
		return sin(10*x*x);
	}
	double exact_rhs(Point p) const override{
		double x = p.x();
		return 400*x*x*sin(10*x*x) - 20*cos(10*x*x);
	}
	ITestPoisson1FemWorker(const IGrid& grid, const FemAssembler& fem): ITestPoissonFemWorker(grid, fem){ }
};

///////////////////////////////////////////////////////////////////////////////
// TestPoissonLinearSegmentWorker
///////////////////////////////////////////////////////////////////////////////
struct TestPoissonLinearSegmentWorker: public ITestPoisson1FemWorker{
	TestPoissonLinearSegmentWorker(const IGrid& grid): ITestPoisson1FemWorker(grid, build_fem(grid)){ }
	static FemAssembler build_fem(const IGrid& grid);
};

FemAssembler TestPoissonLinearSegmentWorker::build_fem(const IGrid& grid){
	size_t n_bases = grid.n_points();
	std::vector<FemElement> elements;
	std::vector<std::vector<size_t>> tab_elem_basis;

	// elements
	for (size_t icell=0; icell < grid.n_cells(); ++icell){
		std::vector<size_t> ipoints = grid.tab_cell_point(icell);
		Point p0 = grid.point(ipoints[0]);
		Point p1 = grid.point(ipoints[1]);
		
		auto geom = std::make_shared<SegmentLinearGeometry>(p0, p1);
		auto basis = std::make_shared<SegmentLinearBasis>();
		auto integrals = std::make_shared<SegmentLinearIntegrals>(geom->jacobi({}));
		FemElement elem{geom, basis, integrals};

		elements.push_back(elem);
		tab_elem_basis.push_back(ipoints);
	}

	return FemAssembler(n_bases, elements, tab_elem_basis);
}
}

///////////////////////////////////////////////////////////////////////////////
// [poisson1-fem-linsegm]
///////////////////////////////////////////////////////////////////////////////
TEST_CASE("Poisson-fem 1D solver, linear segment elements", "[poisson1-fem-linsegm]"){
	std::cout << std::endl << "--- cfd25_test [poisson1-fem-linsegm] --- " << std::endl;
	Grid1D grid(0, 1, 10);
	TestPoissonLinearSegmentWorker worker(grid);
	double nrm = worker.solve();
	worker.save_vtk("poisson1_fem.vtk");
	std::cout << grid.n_cells() << " " << nrm << std::endl;
	CHECK(nrm == Approx(0.138156).margin(1e-6));
}
