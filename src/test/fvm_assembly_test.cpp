#include "cfd25_test.hpp"
#include "cfd/geom/point.hpp"
#include "cfd/grid/regular_grid2d.hpp"
#include "cfd/grid/unstructured_grid2d.hpp"
#include "cfd/fvm/fvm_assembler.hpp"
#include "cfd/grid/vtk.hpp"
#include "utils/filesystem.hpp"

using namespace cfd;

namespace{

double func(Point p){
	double x = p.x();
	double y = p.y();
	return cos(10*x) + sin(5*x*y) + cos(10*y);
}

double dfdx(Point p){
	double x = p.x();
	double y = p.y();
	return -10*sin(10*x) + 5*y*cos(5*x*y);
}

double dfdy(Point p){
	double x = p.x();
	double y = p.y();
	return 5*x*cos(5*x*y) - 10*sin(10*y);
}

}

TEST_CASE("Fvm gradient", "[fvm-gradient]"){
	std::shared_ptr<IGrid> grid;
	int test_no;
	SECTION("Regular grid"){
		test_no = 0;
		grid = std::make_shared<RegularGrid2D>(0, 1, 0, 1, 20, 20);
	}
	SECTION("Unstructured grid"){
		test_no = 1;
		std::string fn = test_directory_file("tetragrid_500.vtk");
		grid = std::make_shared<UnstructuredGrid2D>(UnstructuredGrid2D::vtk_read(fn, true));
	}

	FvmExtendedCollocations collocations(*grid);
	std::vector<double> f;
	std::vector<Vector> exact_grad_f;
	for (Point p: collocations.points){
		f.push_back(func(p));
	}
	for (size_t i=0; i<grid->n_cells(); ++i){
		Point p = grid->cell_center(i);
		exact_grad_f.push_back(Vector{dfdx(p), dfdy(p)});
	}
	std::vector<Vector> grad_f = FvmCellGradient(*grid, collocations).compute(f);

	double sum = 0;
	for (size_t i=0; i<grid->n_cells(); ++i){
		double diff_x = grad_f[i].x() - exact_grad_f[i].x();
		double diff_y = grad_f[i].y() - exact_grad_f[i].y();
		double diff2 = diff_x*diff_x + diff_y*diff_y;
		sum += grid->cell_volume(i) * diff2;
	}
	double residual = std::sqrt(sum);

	switch (test_no){
		case 0: CHECK(residual == Approx(0.855929).margin(1e-6)); break;
		case 1: CHECK(residual == Approx(0.903186).margin(1e-6)); break;
	}
}

TEST_CASE("Fvm linform 2d", "[fvm-linform-2d]"){
	std::string fn = test_directory_file("tetragrid_500.vtk");
	auto grid = UnstructuredGrid2D::vtk_read(fn, true);

	auto f1 = FvmFacesDn(grid);
	auto f2 = FvmLinformFacesDn(grid);

	auto m1 = f1.linear_combination(34);
	auto m2 = f2.linear_combination(34);

	std::map<size_t, double> s2;
	for (auto it: m2){
		s2.insert(it);
	}

	CHECK(m1.size() == s2.size());
	for (auto it: m1){
		CHECK(s2[it.first] == Approx(it.second).margin(1e-8));
	}

	bool passed = true;
	for (size_t iface = 0; iface < grid.n_faces(); ++iface){
		auto m2 = f2.linear_combination(iface);
		if (m2.size() != 4) passed = false;
		if (m2[0].second >= 0.0) passed = false;
		if (m2[1].second <= 0) passed = false;
	}
	CHECK(passed);
}

TEST_CASE("a", "[.a]"){
	RegularGrid2D grid(0, 1, 0, 1, 10, 10);

	FvmLinformFacesDn faces_dn(grid);
	for (size_t iface=0; iface<grid.n_faces(); ++iface){
		auto m2 = faces_dn.linear_combination(iface);
		//auto ed = grid.tab_face_cell(iface);
		//std::cout << iface << " " << ed[0] << "|" << ed[1] << ": ";
		std::cout << m2[0].first << "/" << m2[0].second << ", ";
		std::cout << m2[1].first << "/" << m2[1].second << ", ";
		std::cout << m2[2].first << "/" << m2[2].second << ", ";
		std::cout << m2[3].first << "/" << m2[3].second;
		std::cout << std::endl;
	}
}
