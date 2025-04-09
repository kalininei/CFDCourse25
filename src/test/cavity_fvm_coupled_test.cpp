#include "cfd25_test.hpp"
#include "cfd/grid/vtk.hpp"
#include "cfd/grid/regular_grid2d.hpp"
#include "cfd/mat/lodmat.hpp"
#include "cfd/mat/sparse_matrix_solver.hpp"
#include "cfd/debug/printer.hpp"
#include "utils/vecmat.hpp"
#include "cfd/debug/tictoc.hpp"
#include "cfd/debug/saver.hpp"
#include "cfd/fvm/fvm_assembler.hpp"
#include "utils/filesystem.hpp"
#include "cfd/grid/unstructured_grid2d.hpp"
#include "cfd/fvm/fvm_dpdn_boundary.hpp"
#include <list>

using namespace cfd;

namespace{

double MC(double r){
	return std::max(0.0, std::min(std::min(2.0*r, (1+r)/2.0), 2.0));
}

double tvd_upwind_weight(
		const double* u,
		const std::vector<Vector>& grad_u,
		Vector cij, size_t i, size_t j){

	// treat boundary: take boundary value
	size_t n_cells = grad_u.size();
	if (i >= n_cells){
		return 1.0;
	} else if (j >= n_cells){
		return 0.0;
	}

	// internals
	double dudc = dot_product(grad_u[i], cij);
	double up = u[j] - 2 * dudc;
	double denum = u[i] - u[j];
	if (denum == 0) denum = 1e-12;
	double r = (u[i] - up) / denum;
	double phi = MC(r);
	return 1 - phi/2;
}

}

struct CavityFvmCoupledWorker{
	CavityFvmCoupledWorker(const IGrid& grid, double Re, double tau);
	void initialize_saver(std::string stem);
	double step();
	void save_current_fields(size_t iter);

	size_t vec_size() const{
		return _collocations.size();
	}
private:
	const IGrid& _grid;
	const double _Re;
	const double _tau;
	const FvmExtendedCollocations _collocations;
	const FvmFacesDn _dfdn_computer;
	const GaussLinearFvmCellGradient _grad_computer;

	struct BoundaryInfo{
		std::vector<size_t> bnd_colloc;
		std::vector<size_t> bnd_colloc_u0;
		std::vector<size_t> bnd_colloc_u1;
	};
	BoundaryInfo _boundary_info;

	std::vector<double> _un_face;
	std::vector<Vector> _grad_p;

	CsrMatrix _mat;
	std::vector<double> _rhs;
	std::vector<double> _uvp;

	double* _u;
	double* _v;
	double* _p;

	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer;

	void to_next_iteration();
	void gather_boundary_collocations();
	std::vector<double> compute_un_rhie_chow() const;
	void u_equation(CsrMatrix&, std::vector<double>&) const;
	void v_equation(CsrMatrix&, std::vector<double>&) const;
	void continuity_equation(CsrMatrix&, std::vector<double>&) const;
	void assemble_slae();
};

CavityFvmCoupledWorker::CavityFvmCoupledWorker(const IGrid& grid, double Re, double tau):
	_grid(grid),
	_Re(Re),
	_tau(tau),
	_collocations(grid),
	_dfdn_computer(grid, _collocations),
	_grad_computer(grid, _collocations)
{
	gather_boundary_collocations();

	_uvp = std::vector<double>(3*vec_size(), 0);
	_rhs = std::vector<double>(3*vec_size(), 0);
	_u = _uvp.data();
	_v = _uvp.data() + vec_size();
	_p = _uvp.data() + 2*vec_size();
	_un_face = std::vector<double>(_grid.n_faces(), 0);
	_grad_p = std::vector<Vector>(_grid.n_cells(), {0, 0, 0});
	to_next_iteration();
}

void CavityFvmCoupledWorker::gather_boundary_collocations(){
	BoundaryInfo& bi = _boundary_info;

	std::list<std::pair<size_t, size_t>> colloc_faces;
	for (size_t icolloc: _collocations.face_collocations){
		size_t iface = _collocations.face_index(icolloc);
		bi.bnd_colloc.push_back(icolloc);
		if (std::abs(_grid.face_center(iface).y() - 1) < 1e-6){
			bi.bnd_colloc_u1.push_back(icolloc);
		} else {
			bi.bnd_colloc_u0.push_back(icolloc);
		}
	}
}

void CavityFvmCoupledWorker::initialize_saver(std::string stem){
	_writer.reset(new VtkUtils::TimeSeriesWriter(stem));
};

void CavityFvmCoupledWorker::to_next_iteration(){
	assemble_slae();
};

void CavityFvmCoupledWorker::u_equation(CsrMatrix& B, std::vector<double>& rhs) const{
	LodMatrix B0(vec_size()), B2(vec_size());
	std::fill(rhs.begin(), rhs.end(), 0.0);
	std::vector<Vector> grad_u = _grad_computer.compute(_u);

	// cell loop
	for (size_t icell=0; icell<_grid.n_cells(); ++icell){
		double vol = _grid.cell_volume(icell);
		// nonstat
		B0.add_value(icell, icell, vol);
		rhs[icell] += vol * _u[icell];
	}
	// face loop
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);
		double un = _un_face[iface];

		// - tau/Re * Laplace(u)
		for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
			size_t column = iter.first;
			double coef = -_tau/_Re * area * iter.second;
			B0.add_value(negative_colloc, column, coef);
			B0.add_value(positive_colloc, column, -coef);
		}
		
		// + nonlinear tvd convection
		{
			size_t i = negative_colloc; // upwind cell
			size_t j = positive_colloc; // downwind cell
			double coef = _tau * area * un;
			if (un < 0){
				std::swap(i, j);
				coef *= -1;
			};
			Vector cij = _collocations.points[j] - _collocations.points[i];
			double wi = tvd_upwind_weight(_u, grad_u, cij, i, j);
			double wj = 1 - wi;
			B0.add_value(i, i,  wi*coef);
			B0.add_value(i, j,  wj*coef);
			B0.add_value(j, j, -wj*coef);
			B0.add_value(j, i, -wi*coef);
		}
		// + dp/dx
		{
			// _tau * area * nx / 2
			double value = _tau * area * _grid.face_normal(iface).x() * 0.5;

			if (_collocations.is_boundary_colloc(negative_colloc)){
				B2.add_value(positive_colloc, negative_colloc, -2*value);
			} else if (_collocations.is_boundary_colloc(positive_colloc)){
				B2.add_value(negative_colloc, positive_colloc, 2*value);
			} else {
				B2.add_value(negative_colloc, negative_colloc, value);
				B2.add_value(negative_colloc, positive_colloc, value);
				B2.add_value(positive_colloc, negative_colloc, -value);
				B2.add_value(positive_colloc, positive_colloc, -value);
			}
		}
	}

	// boundary conditions
	for (size_t icolloc: _boundary_info.bnd_colloc){
		B0.set_unit_row(icolloc);
		B2.remove_row(icolloc);
		rhs[icolloc] = 0;
	}
	for (size_t icolloc: _boundary_info.bnd_colloc_u1){
		rhs[icolloc] = 1;
	}

	// block assemble
	B = assemble_block_matrix(vec_size(), vec_size(), { {&B0, nullptr, &B2} });
};

void CavityFvmCoupledWorker::v_equation(CsrMatrix& B, std::vector<double>& rhs) const{
	LodMatrix B1(vec_size()), B2(vec_size());
	std::fill(rhs.begin(), rhs.end(), 0.0);
	std::vector<Vector> grad_v = _grad_computer.compute(_v);

	// cell loop
	for (size_t icell=0; icell<_grid.n_cells(); ++icell){
		double vol = _grid.cell_volume(icell);
		// nonstat
		B1.add_value(icell, icell, vol);
		rhs[icell] += vol * _v[icell];
	}
	// face loop
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);
		double un = _un_face[iface];

		// - tau/Re * Laplace(u)
		for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
			size_t column = iter.first;
			double coef = -_tau/_Re * area * iter.second;
			B1.add_value(negative_colloc, column, coef);
			B1.add_value(positive_colloc, column, -coef);
		}
		
		// + nonlinear tvd convection
		{
			size_t i = negative_colloc; // upwind cell
			size_t j = positive_colloc; // downwind cell
			double coef = _tau * area * un;
			if (un < 0){
				std::swap(i, j);
				coef *= -1;
			};
			Vector cij = _collocations.points[j] - _collocations.points[i];
			double wi = tvd_upwind_weight(_v, grad_v, cij, i, j);
			double wj = 1 - wi;
			B1.add_value(i, i,  wi*coef);
			B1.add_value(i, j,  wj*coef);
			B1.add_value(j, j, -wj*coef);
			B1.add_value(j, i, -wi*coef);
		}
		// + dp/dy
		{
			// _tau * area * ny / 2
			double value = _tau * area * _grid.face_normal(iface).y() * 0.5;

			if (_collocations.is_boundary_colloc(negative_colloc)){
				B2.add_value(positive_colloc, negative_colloc, -2*value);
			} else if (_collocations.is_boundary_colloc(positive_colloc)){
				B2.add_value(negative_colloc, positive_colloc, 2*value);
			} else {
				B2.add_value(negative_colloc, negative_colloc, value);
				B2.add_value(negative_colloc, positive_colloc, value);
				B2.add_value(positive_colloc, negative_colloc, -value);
				B2.add_value(positive_colloc, positive_colloc, -value);
			}
		}
	}

	// boundary conditions
	for (size_t icolloc: _boundary_info.bnd_colloc){
		B1.set_unit_row(icolloc);
		B2.remove_row(icolloc);
		rhs[icolloc] = 0;
	}

	// block assemble
	B = assemble_block_matrix(vec_size(), vec_size(), { {nullptr, &B1, &B2} });
};

void CavityFvmCoupledWorker::continuity_equation(CsrMatrix& B, std::vector<double>& rhs) const{
	LodMatrix B0(vec_size()), B1(vec_size()), B2(vec_size());
	std::fill(rhs.begin(), rhs.end(), 0.0);

	// face loop
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);
		Vector normal = _grid.face_normal(iface);
		double nx = normal[0];
		double ny = normal[1];

		// (u_i + u_j)/2
		{
			double value_x = 0.5 * area * nx;
			double value_y = 0.5 * area * ny;

			if (_collocations.is_boundary_colloc(negative_colloc)){
				B0.add_value(positive_colloc, negative_colloc, -2*value_x);
				B1.add_value(positive_colloc, negative_colloc, -2*value_y);
			} else if (_collocations.is_boundary_colloc(positive_colloc)){
				B0.add_value(negative_colloc, positive_colloc, 2*value_x);
				B1.add_value(negative_colloc, positive_colloc, 2*value_y);
			} else {
				B0.add_value(negative_colloc, negative_colloc, value_x);
				B0.add_value(negative_colloc, positive_colloc, value_x);
				B0.add_value(positive_colloc, negative_colloc, -value_x);
				B0.add_value(positive_colloc, positive_colloc, -value_x);

				B1.add_value(negative_colloc, negative_colloc, value_y);
				B1.add_value(negative_colloc, positive_colloc, value_y);
				B1.add_value(positive_colloc, negative_colloc, -value_y);
				B1.add_value(positive_colloc, positive_colloc, -value_y);
			}
		}
	
		if (_collocations.is_internal_colloc(negative_colloc) && _collocations.is_internal_colloc(positive_colloc)){
			// tau * (dp/dn)_ij
			for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
				size_t column = iter.first;
				double coef = -_tau * area * iter.second;
				B2.add_value(negative_colloc, column, coef);
				B2.add_value(positive_colloc, column, -coef);
			}

			// 0.5 * tau * ( (dp/dn)_i + (dp/dn)_ j)
			double coeff = -_tau * area;
			double dpdn = 0.5*(
				dot_product(_grad_p[negative_colloc], normal) +
				dot_product(_grad_p[positive_colloc], normal));
			rhs[negative_colloc] += coeff * dpdn;
			rhs[positive_colloc] += -coeff * dpdn;
		}
	}

	// boundary condition: dp/dn = 0
	for (size_t icolloc: _boundary_info.bnd_colloc){
		B0.remove_row(icolloc);
		B1.remove_row(icolloc);
		B2.remove_row(icolloc);
		size_t iface = _collocations.face_index(icolloc);
		double area = _grid.face_area(iface);
		double sgn = (_collocations.tab_face_colloc(iface)[1] == icolloc) ? 1.0 : -1.0;

		for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
			size_t column = iter.first;
			double coef = sgn * area * iter.second;
			B2.add_value(icolloc, column, coef);
		}
		rhs[icolloc] = 0;
	};
	// p=0 at icell=0
	B0.remove_row(0);
	B1.remove_row(0);
	B2.set_unit_row(0);
	rhs[0] = 0;

	// block assemble
	B = assemble_block_matrix(vec_size(), vec_size(), { {&B0, &B1, &B2} });

};

void CavityFvmCoupledWorker::assemble_slae(){
	std::vector<double> rhs_u(vec_size(), 0);
	std::vector<double> rhs_v(vec_size(), 0);
	std::vector<double> rhs_continuity(vec_size(), 0);
	CsrMatrix Bu, Bv, Bcontinuity;

	u_equation(Bu, rhs_u);
	v_equation(Bv, rhs_v);
	continuity_equation(Bcontinuity, rhs_continuity);

	_mat = assemble_block_matrix(vec_size(), 3*vec_size(), { {&Bu}, {&Bv}, {&Bcontinuity} });

	std::copy(rhs_u.begin(), rhs_u.end(), _rhs.begin());
	std::copy(rhs_v.begin(), rhs_v.end(), _rhs.begin()+vec_size());
	std::copy(rhs_continuity.begin(), rhs_continuity.end(), _rhs.begin()+2*vec_size());
}

double CavityFvmCoupledWorker::step(){
	std::vector<double> uvp_old(_uvp);
	AmgcMatrixSolver::solve_slae(_mat, _rhs, _uvp);

	// d/dt
	double dudt = 0, dvdt=0, dpdt=0;
	for (size_t i=0; i<vec_size(); ++i){
		dudt = std::max(dudt, std::abs(uvp_old[i] - _uvp[i]));
		dvdt = std::max(dvdt, std::abs(uvp_old[i+vec_size()] - _uvp[i+vec_size()]));
		dpdt = std::max(dvdt, std::abs(uvp_old[i+2*vec_size()] - _uvp[i+2*vec_size()]));
	}
	dudt /= _tau;
	dvdt /= _tau;
	dpdt /= _tau;

	// additional values
	_grad_p = _grad_computer.compute(_p);
	_un_face = compute_un_rhie_chow();

	to_next_iteration();
	return std::max(std::max(dudt, dvdt), dpdt);
}

void CavityFvmCoupledWorker::save_current_fields(size_t iter){
	if (_writer){
		std::string filepath = _writer->add(iter);
		_grid.save_vtk(filepath);
		const size_t nc = _grid.n_cells();
		VtkUtils::add_cell_data(std::vector<double>(_p, _p + nc),
				"pressure", filepath, _grid.n_cells());
		VtkUtils::add_cell_vector(std::vector<double>(_u, _u + nc),
				std::vector<double>(_v, _v + nc),
				"velocity", filepath, _grid.n_cells());
	}
}

std::vector<double> CavityFvmCoupledWorker::compute_un_rhie_chow() const{

	std::vector<double> ret(_grid.n_faces());
	std::vector<double> dpdn_face = _dfdn_computer.compute(_p);

	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		size_t ci = _grid.tab_face_cell(iface)[0];
		size_t cj = _grid.tab_face_cell(iface)[1];
		if (ci == INVALID_INDEX || cj == INVALID_INDEX){
			ret[iface] = 0;
		} else {
			// Rhie-Chow interpolation
			Vector normal = _grid.face_normal(iface);
			Vector uvec_i = Vector(_u[ci], _v[ci]);
			Vector uvec_j = Vector(_u[cj], _v[cj]);
			
			double u_i = dot_product(uvec_i, normal);
			double u_j = dot_product(uvec_j, normal);
			double dpdn_i = dot_product(_grad_p[ci], normal);
			double dpdn_j = dot_product(_grad_p[cj], normal);
			double dpdn_ij = dpdn_face[iface];

			ret[iface] = 0.5*(u_i + u_j)
			           + 0.5*_tau*(dpdn_i + dpdn_j - 2*dpdn_ij);
		}
	}

	return ret;
}

TEST_CASE("Cavity FVM-SIMPLE coupled algorithm", "[cavity-fvm-coupled]"){
	std::cout << std::endl << "--- cfd25_test [cavity-fvm-coupled] --- " << std::endl;

	// problem parameters
	double Re = 100;
	size_t max_it = 10'000;
	double eps = 1e-3;
	double tau = 0.5;

	// worker initialization
	RegularGrid2D grid(0, 1, 0, 1, 30, 30);
	//std::string fn = test_directory_file("tetragrid_500.vtk");
	//UnstructuredGrid2D grid = UnstructuredGrid2D::vtk_read(fn);
	CavityFvmCoupledWorker worker(grid, Re, tau);
	worker.initialize_saver("cavity-fvm-coupled");

	// iterations loop
	size_t it = 0;
	for (it=1; it < max_it; ++it){
		double nrm = worker.step();

		// print norm and friction force 
		std::cout << it << " " << nrm << std::endl;

		// export solution to vtk
		//worker.save_current_fields(it);

		// break if residual is low enough
		if (nrm < eps){
			break;
		}
	}
	CHECK(it == 29);
}
