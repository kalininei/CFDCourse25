#include "cfd25_test.hpp"
#include "test/utils/filesystem.hpp"
#include "cfd/grid/grid1d.hpp"
#include "cfd/mat/csrmat.hpp"
#include "cfd/mat/sparse_matrix_solver.hpp"
#include "cfd/fem/fem_assembler.hpp"
#include "cfd/fem/fem_numeric_integrals.hpp"
#include "cfd/fem/elem1d/segment_linear.hpp"
#include "cfd/fem/elem2d/triangle_linear.hpp"
#include "cfd/fem/elem2d/triangle_quadratic.hpp"
#include "cfd/fem/elem2d/triangle_cubic.hpp"
#include "cfd/fem/elem2d/quadrangle_linear.hpp"
#include "cfd/fem/elem2d/quadrangle_quadratic.hpp"
#include "cfd/fem/fem_sorted_cell_info.hpp"
#include "cfd/numeric_integration/square_quadrature.hpp"
#include "cfd/numeric_integration/triangle_quadrature.hpp"
#include "cfd/grid/vtk.hpp"
#include "cfd/grid/unstructured_grid2d.hpp"
#include "cfd/grid/regular_grid2d.hpp"
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

////////////////////////////////////////////////////////////////////////////////
// ITestPoisson2FemWorker
////////////////////////////////////////////////////////////////////////////////
struct ITestPoisson2FemWorker: public ITestPoissonFemWorker{
	double exact_solution(Point p) const override{
		double x = p.x();
		double y = p.y();
		return cos(10*x*x)*sin(10*y) + sin(10*x*x)*cos(10*x);
	}
	double exact_rhs(Point p) const override{
		double x = p.x();
		double y = p.y();
		return (20*sin(10*x*x)+(400*x*x+100)*cos(10*x*x))*sin(10*y)
				+(400*x*x+100)*cos(10*x)*sin(10*x*x)
				+(400*x*sin(10*x)-20*cos(10*x))*cos(10*x*x);
	}
	ITestPoisson2FemWorker(const IGrid& grid, const FemAssembler& fem): ITestPoissonFemWorker(grid, fem){ }
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

///////////////////////////////////////////////////////////////////////////////
// TestPoissonLinearTriangleWorker
///////////////////////////////////////////////////////////////////////////////
struct TestPoissonLinearTriangleWorker: public ITestPoisson2FemWorker{
	static FemAssembler build_fem(const IGrid& grid);
	TestPoissonLinearTriangleWorker(const IGrid& grid): ITestPoisson2FemWorker(grid, build_fem(grid)){ }
};

FemAssembler TestPoissonLinearTriangleWorker::build_fem(const IGrid& grid){
	size_t n_bases = grid.n_points();
	std::vector<FemElement> elements;
	std::vector<std::vector<size_t>> tab_elem_basis;

	// elements
	for (size_t icell=0; icell < grid.n_cells(); ++icell){
		std::vector<size_t> ipoints = grid.tab_cell_point(icell);
		Point p0 = grid.point(ipoints[0]);
		Point p1 = grid.point(ipoints[1]);
		Point p2 = grid.point(ipoints[2]);
		
		auto geom = std::make_shared<TriangleLinearGeometry>(p0, p1, p2);
		auto basis = std::make_shared<TriangleLinearBasis>();
		JacobiMatrix jac = geom->jacobi({0, 0});
		auto integrals = std::make_shared<TriangleLinearIntegrals>(jac);
		FemElement elem{geom, basis, integrals};

		elements.push_back(elem);
		tab_elem_basis.push_back(ipoints);
	}

	return FemAssembler(n_bases, elements, tab_elem_basis);
}

TEST_CASE("Poisson-fem 2D solver, triangles", "[poisson2-fem-tri]"){
	std::cout << std::endl << "--- cfd25_test [poisson2-fem-tri] --- " << std::endl;

	std::string grid_fn = test_directory_file("trigrid_500.vtk");
	UnstructuredGrid2D grid = UnstructuredGrid2D::vtk_read(grid_fn, true);
	TestPoissonLinearTriangleWorker worker(grid);
	double nrm = worker.solve();
	worker.save_vtk("poisson2_fem.vtk");
	std::cout << grid.n_cells() << " " << nrm << std::endl;
	CHECK(nrm == Approx(0.0638327072).margin(1e-6));
}

///////////////////////////////////////////////////////////////////////////////
// TestPoissonBilinearQuadrangleWorker
///////////////////////////////////////////////////////////////////////////////
struct TestPoissonBilinearQuadrangleWorker: public ITestPoisson2FemWorker{
	static FemAssembler build_fem(const IGrid& grid);
	TestPoissonBilinearQuadrangleWorker(const IGrid& grid): ITestPoisson2FemWorker(grid, build_fem(grid)){ }
};

FemAssembler TestPoissonBilinearQuadrangleWorker::build_fem(const IGrid& grid){
	size_t n_bases = grid.n_points();
	std::vector<FemElement> elements;
	std::vector<std::vector<size_t>> tab_elem_basis;

	// elements
	for (size_t icell=0; icell < grid.n_cells(); ++icell){
		std::vector<size_t> ipoints = grid.tab_cell_point(icell);
		Point p0 = grid.point(ipoints[0]);
		Point p1 = grid.point(ipoints[1]);
		Point p2 = grid.point(ipoints[2]);
		Point p3 = grid.point(ipoints[3]);
		
		auto geom = std::make_shared<QuadrangleLinearGeometry>(p0, p1, p2, p3);
		auto basis = std::make_shared<QuadrangleLinearBasis>();
		const Quadrature* quadrature = quadrature_square_gauss2();
		auto integrals = std::make_shared<NumericElementIntegrals>(quadrature, geom, basis);
		FemElement elem{geom, basis, integrals};

		elements.push_back(FemElement{geom, basis, integrals});
		tab_elem_basis.push_back(ipoints);
	}

	return FemAssembler(n_bases, elements, tab_elem_basis);
}

TEST_CASE("Poisson-fem 2D solver, quadrangles", "[poisson2-fem-quad]"){
	std::cout << std::endl << "--- cfd25_test [poisson2-fem-quad] --- " << std::endl;
	RegularGrid2D grid(0, 1, 0, 1, 10, 10);
	TestPoissonBilinearQuadrangleWorker worker(grid);
	double nrm = worker.solve();
	worker.save_vtk("poisson2_fem.vtk");
	std::cout << grid.n_cells() << " " << nrm << std::endl;
	CHECK(nrm == Approx(0.1181241688).margin(1e-6));
}

///////////////////////////////////////////////////////////////////////////////
// TestPoissonQuadraticWorker
///////////////////////////////////////////////////////////////////////////////
struct TestPoissonQuadraticWorker: public ITestPoisson2FemWorker{
	std::vector<size_t> dirichlet_bases() const override{
		std::vector<size_t> ret = _grid.boundary_points();
		for (size_t iface: _grid.boundary_faces()){
			ret.push_back(_grid.n_points() + iface);
		}
		return ret;
	}
	static FemAssembler build_fem(const IGrid& grid);
	TestPoissonQuadraticWorker(const IGrid& grid): ITestPoisson2FemWorker(grid, build_fem(grid)){ }

};

FemAssembler TestPoissonQuadraticWorker::build_fem(const IGrid& grid){
	std::vector<FemElement> elements;
	std::vector<std::vector<size_t>> tab_elem_basis;

	// elements
	size_t n_quad_cells = 0;
	for (size_t icell=0; icell < grid.n_cells(); ++icell){
		PolygonElementInfo cell_info(grid, icell);
		if (cell_info.n_points() == 3){
			Point p0 = grid.point(cell_info.ipoints[0]);
			Point p1 = grid.point(cell_info.ipoints[1]);
			Point p2 = grid.point(cell_info.ipoints[2]);
			
			auto geom = std::make_shared<TriangleLinearGeometry>(p0, p1, p2);
			auto basis = std::make_shared<TriangleQuadraticBasis>();
			auto integrals = std::make_shared<NumericElementIntegrals>(quadrature_triangle_gauss4(), geom, basis);

			elements.push_back(FemElement{geom, basis, integrals});
			tab_elem_basis.push_back({
				cell_info.ipoints[0],
				cell_info.ipoints[1],
				cell_info.ipoints[2],
				grid.n_points() + cell_info.ifaces[0],
				grid.n_points() + cell_info.ifaces[1],
				grid.n_points() + cell_info.ifaces[2]
			});
		} else if (cell_info.n_points() == 4){
			Point p0 = grid.point(cell_info.ipoints[0]);
			Point p1 = grid.point(cell_info.ipoints[1]);
			Point p2 = grid.point(cell_info.ipoints[2]);
			Point p3 = grid.point(cell_info.ipoints[3]);
			
			auto geom = std::make_shared<QuadrangleLinearGeometry>(p0, p1, p2, p3);
			auto basis = std::make_shared<QuadrangleQuadraticBasis>();
			auto integrals = std::make_shared<NumericElementIntegrals>(quadrature_square_gauss4(), geom, basis);

			elements.push_back(FemElement{geom, basis, integrals});
			tab_elem_basis.push_back({
				cell_info.ipoints[0],
				cell_info.ipoints[1],
				cell_info.ipoints[2],
				cell_info.ipoints[3],
				grid.n_points() + cell_info.ifaces[0],
				grid.n_points() + cell_info.ifaces[1],
				grid.n_points() + cell_info.ifaces[2],
				grid.n_points() + cell_info.ifaces[3],
				grid.n_points() + grid.n_faces() + n_quad_cells
			});
			n_quad_cells++;
		} else {
			throw std::runtime_error("Invalid fem grid");
		}
	}

	size_t n_bases = grid.n_points() + grid.n_faces() + n_quad_cells;

	return FemAssembler(n_bases, elements, tab_elem_basis);
}

TEST_CASE("Poisson-fem 2D solver, quadratic", "[poisson2-fem-quadratic]"){
	std::cout << std::endl << "--- cfd25_test [poisson2-fem-quadratic] --- " << std::endl;

	std::string grid_fn = test_directory_file("tetragrid_500.vtk");
	UnstructuredGrid2D grid = UnstructuredGrid2D::vtk_read(grid_fn);
	TestPoissonQuadraticWorker worker(grid);
	double nrm = worker.solve();
	std::cout << grid.n_cells() << " " << nrm << std::endl;
	CHECK(nrm == Approx(0.0012520815).margin(1e-6));
}

///////////////////////////////////////////////////////////////////////////////
// TestPoissonCubicWorker
///////////////////////////////////////////////////////////////////////////////
struct TestPoissonCubicWorker: public ITestPoisson2FemWorker{
	std::vector<size_t> dirichlet_bases() const override{
		std::vector<size_t> ret = _grid.boundary_points();
		for (size_t iface: _grid.boundary_faces()){
			ret.push_back(_grid.n_points() + 2*iface);
			ret.push_back(_grid.n_points() + 2*iface+1);
		}
		return ret;
	}
	static FemAssembler build_fem(const IGrid& grid);
	TestPoissonCubicWorker(const IGrid& grid): ITestPoisson2FemWorker(grid, build_fem(grid)){ }

};

FemAssembler TestPoissonCubicWorker::build_fem(const IGrid& grid){
	std::vector<FemElement> elements;
	std::vector<std::vector<size_t>> tab_elem_basis;

	// elements
	for (size_t icell=0; icell < grid.n_cells(); ++icell){
		PolygonElementInfo cell_info(grid, icell);

		if (cell_info.n_points() == 3){
			Point p0 = grid.point(cell_info.ipoints[0]);
			Point p1 = grid.point(cell_info.ipoints[1]);
			Point p2 = grid.point(cell_info.ipoints[2]);
			
			auto geom = std::make_shared<TriangleLinearGeometry>(p0, p1, p2);
			auto basis = std::make_shared<TriangleCubicBasis>();
			auto integrals = std::make_shared<NumericElementIntegrals>(quadrature_triangle_gauss4(), geom, basis);

			elements.push_back(FemElement{geom, basis, integrals});

			size_t bas0 = cell_info.ipoints[0];
			size_t bas1 = cell_info.ipoints[1];
			size_t bas2 = cell_info.ipoints[2];
			size_t bas3 = grid.n_points() + 2*cell_info.ifaces[0];
			size_t bas4 = grid.n_points() + 2*cell_info.ifaces[0] + 1;
			size_t bas5 = grid.n_points() + 2*cell_info.ifaces[1];
			size_t bas6 = grid.n_points() + 2*cell_info.ifaces[1] + 1;
			size_t bas7 = grid.n_points() + 2*cell_info.ifaces[2];
			size_t bas8 = grid.n_points() + 2*cell_info.ifaces[2] + 1;
			size_t bas9 = grid.n_points() + 2*grid.n_faces() + icell;
			if (cell_info.is_face_reverted[0]) std::swap(bas3, bas4);
			if (cell_info.is_face_reverted[1]) std::swap(bas5, bas6);
			if (cell_info.is_face_reverted[2]) std::swap(bas7, bas8);

			tab_elem_basis.push_back({
				bas0, bas1, bas2,
				bas3, bas4, bas5, bas6, bas7, bas8,
				bas9
			});
		} else {
			throw std::runtime_error("Invalid fem grid");
		}
	}

	size_t n_bases = grid.n_points() + 2*grid.n_faces() + grid.n_cells();
	return FemAssembler(n_bases, elements, tab_elem_basis);
}

TEST_CASE("Poisson-fem 2D solver, cubic", "[poisson2-fem-cubic]"){
	std::cout << std::endl << "--- cfd25_test [poisson2-fem-cubic] --- " << std::endl;

	std::string grid_fn = test_directory_file("trigrid_500.vtk");
	UnstructuredGrid2D grid = UnstructuredGrid2D::vtk_read(grid_fn);
	TestPoissonCubicWorker worker(grid);
	double nrm = worker.solve();
	std::cout << grid.n_cells() << " " << nrm << std::endl;
	CHECK(nrm == Approx(0.0005724831).margin(1e-6));
}
