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
#include <iomanip>
#include <list>

using namespace cfd;

namespace{

// => coef * (M - diag(M))* u
std::vector<double> nodiag_mult(double coef, const CsrMatrix& mat, const std::vector<double>& u){
	std::vector<double> ret(u.size(), 0);
	for (size_t i=0; i<ret.size(); ++i){
		for (size_t a=mat.addr()[i]; a < mat.addr()[i+1]; ++a){
			if (mat.cols()[a] != i){
				ret[i] += coef * mat.vals()[a] * u[mat.cols()[a]]; 
			}
		}
	}
	return ret;
}

};

struct CylinderFvmPimpleWorker{
	CylinderFvmPimpleWorker(const IGrid& grid, double Re, double E, double timestep, size_t n_piso);
	void initialize_saver(std::string stem, double save_time_step);

	double step();
	double to_next_time_step();
	void save_current_fields(double time) const;

	size_t vec_size() const{
		return _collocations.size();
	}
private:
	const IGrid& _grid;
	const double _Re;
	const double _tau;
	const double _alpha_p;
	const double _time_step;
	const size_t _n_piso;
	const FvmExtendedCollocations _collocations;
	const FvmFacesDn _dfdn_computer;
	const std::shared_ptr<IFvmCellGradient> _grad_computer;

	struct BoundaryInfo{
		std::vector<size_t> all;
		std::vector<size_t> cyl;
		std::vector<size_t> input;
		std::vector<size_t> output;
		std::vector<size_t> sym;
	};
	BoundaryInfo _boundary_info;

	std::vector<double> _p;
	std::vector<double> _u;
	std::vector<double> _v;
	std::vector<double> _un_face;
	std::vector<double> _u_old;
	std::vector<double> _v_old;

	AmgcMatrixSolver _p_prime_solver;
	AmgcMatrixSolver _uv_solver;

	CsrMatrix _mat_uv;
	std::vector<double> _rhs_u;
	std::vector<double> _rhs_v;
	double _d;

	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer;

	double to_next_iteration();
	void gather_boundary_collocations();
	void assemble_p_prime_solver();
	CsrMatrix assemble_uv_lhs(double coef_u, double coef_conv, double coef_diff) const;
	void assemble_uv_slae();

	std::vector<double> compute_u_star();
	std::vector<double> compute_v_star();
	std::vector<double> compute_un_face(
			const std::vector<double>& u, const std::vector<double>& v, double bval,
			bool rhie_chow, const std::vector<double>& p={});
	std::vector<double> compute_un_face_prime(const std::vector<double>& p);
	std::vector<double> compute_p_prime(const std::vector<double>& un_star_face);
	std::vector<double> compute_u_prime(const std::vector<Vector>& grad_p_prime);
	std::vector<double> compute_v_prime(const std::vector<Vector>& grad_p_prime);

	static double compute_tau(const IGrid& grid, double Re, double E);
};

double CylinderFvmPimpleWorker::compute_tau(const IGrid& grid, double Re, double E){
	double h2 = grid.cell_volume(0);
	for (size_t i=1; i<grid.n_cells(); ++i){
		h2 = std::min(h2, grid.cell_volume(i));
	}
	return E*Re*h2/4.0;
}

CylinderFvmPimpleWorker::CylinderFvmPimpleWorker(const IGrid& grid, double Re, double E, double time_step, size_t n_piso):
	_grid(grid),
	_Re(Re),
	_tau(compute_tau(grid, Re, E)),
	_alpha_p(1.0/(1.0 + _tau/time_step + E)),
	_time_step(time_step),
	_n_piso(n_piso),
	_collocations(grid),
	_dfdn_computer(grid, _collocations),
	_grad_computer(std::make_shared<GaussLinearFvmCellGradient>(grid, _collocations))
{

	_d = 1.0/(1 + _tau/time_step + E);
	gather_boundary_collocations();
	assemble_p_prime_solver();

	_u = std::vector<double>(vec_size(), 1);
	_v = std::vector<double>(vec_size(), 0);
	_p = std::vector<double>(vec_size(), 0);
	_un_face = compute_un_face(_u, _v, 1.0, false);
	to_next_time_step();
}

void CylinderFvmPimpleWorker::gather_boundary_collocations(){
	double xmin = _grid.point(0).x(); double xmax = _grid.point(0).x();
	double ymin = _grid.point(0).y(); double ymax = _grid.point(0).y();
	for (size_t i=1; i<_grid.n_points(); ++i){
		Point p = _grid.point(i);
		xmin = std::min(xmin, p.x()); ymin = std::min(ymin, p.y());
		xmax = std::max(xmax, p.x()); ymax = std::max(ymax, p.y());
	}

	BoundaryInfo& bi = _boundary_info;
	for (size_t icolloc: _collocations.face_collocations){
		size_t iface = _collocations.face_index(icolloc);
		bi.all.push_back(icolloc);
		Point fc = _grid.face_center(iface);
		if (std::abs(fc.y() - ymin) < 1e-6){
			bi.sym.push_back(icolloc);
		} else if (std::abs(fc.y() - ymax) < 1e-6){
			bi.sym.push_back(icolloc);
		} else if (std::abs(fc.x() - xmin) < 1e-6){
			bi.input.push_back(icolloc);
		} else if (std::abs(fc.x() - xmax) < 1e-6){
			bi.output.push_back(icolloc);
		} else {
			bi.cyl.push_back(icolloc);
		}
	}
}

void CylinderFvmPimpleWorker::initialize_saver(std::string stem, double save_time_step){
	_writer.reset(new VtkUtils::TimeSeriesWriter(stem));
	_writer->set_time_step(save_time_step);
};

double CylinderFvmPimpleWorker::to_next_iteration(){
	assemble_uv_slae();
	// residual vectors
	std::vector<double> res_u = compute_residual_vec(_mat_uv, _rhs_u, _u);
	std::vector<double> res_v = compute_residual_vec(_mat_uv, _rhs_v, _v);
	for (size_t icell=0; icell < _grid.n_cells(); ++icell){
		double coef = 1.0 / _tau / _grid.cell_volume(icell);
		res_u[icell] *= coef;
		res_v[icell] *= coef;
	}
	// norm
	double res = 0;
	for (size_t icell=0; icell < _grid.n_cells(); ++icell){
		res = std::max(res, std::max(res_u[icell], res_v[icell]));
	}
	return res;
};

double CylinderFvmPimpleWorker::step(){
	// ========= SIMPLE STEP
	// Predictor step: U-star
	std::vector<double> u_star = compute_u_star();
	std::vector<double> v_star = compute_v_star();
	std::vector<double> un_star_face = compute_un_face(u_star, v_star, 1.0, true, _p);
	// Pressure correction
	std::vector<double> p_prime = compute_p_prime(un_star_face);
	std::vector<Vector> grad_p_prime = _grad_computer->compute(p_prime);
	// Velocity correction
	std::vector<double> u_prime = compute_u_prime(grad_p_prime);
	std::vector<double> v_prime = compute_v_prime(grad_p_prime);
	std::vector<double> un_prime_face = compute_un_face_prime(p_prime);

	// =========== PISO Step
	std::vector<double> u_new = vector_sum(u_star, 1.0, u_prime);
	std::vector<double> v_new = vector_sum(v_star, 1.0, v_prime);
	std::vector<double> un_face_new = vector_sum(un_star_face, 1.0, un_prime_face);
	std::vector<double> p_new = vector_sum(_p, _alpha_p, p_prime);

	for (size_t k=0; k < _n_piso; ++k){
		// ==== u-tilde = -H*du*u'
		std::vector<double> hu = nodiag_mult(-_d, _mat_uv, u_prime);
		std::vector<double> hv = nodiag_mult(-_d, _mat_uv, v_prime);
		std::vector<double> hn_face = compute_un_face(hu, hv, 0.0, false);
		// pressure piso correction
		p_prime = compute_p_prime(hn_face);
		grad_p_prime = _grad_computer->compute(p_prime);
		// velocity piso correction
		u_prime = compute_u_prime(grad_p_prime);
		v_prime = compute_v_prime(grad_p_prime);
		un_prime_face = compute_un_face_prime(p_prime);

		for (size_t i=0; i<u_new.size(); ++i) u_new[i] += (u_prime[i] + hu[i]);
		for (size_t i=0; i<v_new.size(); ++i) v_new[i] += (v_prime[i] + hv[i]);
		for (size_t i=0; i<un_face_new.size(); ++i) un_face_new[i] += (un_prime_face[i] + hn_face[i]);
		for (size_t i=0; i<p_new.size(); ++i) p_new[i] += _alpha_p * p_prime[i];
	}

	std::swap(u_new, _u);
	std::swap(v_new, _v);
	std::swap(p_new, _p);
	std::swap(un_face_new, _un_face);

	return to_next_iteration();
}

void CylinderFvmPimpleWorker::save_current_fields(double time) const{
	if (_writer){
		std::string filepath = _writer->add(time);
		if (filepath.empty()){
			return;
		}
		_grid.save_vtk(filepath);
		VtkUtils::add_cell_data(_p, "pressure", filepath, _grid.n_cells());
		VtkUtils::add_cell_vector(_u, _v, "velocity", filepath, _grid.n_cells());
	}
}

void CylinderFvmPimpleWorker::assemble_p_prime_solver(){
	LodMatrix mat(vec_size());
	// internal
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);

		for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
			size_t column = iter.first;
			double coef = -_d * area * iter.second;
			mat.add_value(negative_colloc, column, coef);
			mat.add_value(positive_colloc, column, -coef);
		}
	}
	mat.set_unit_row(0);
	_p_prime_solver.set_matrix(mat.to_csr());
}

CsrMatrix CylinderFvmPimpleWorker::assemble_uv_lhs(double coef_u, double coef_conv, double coef_diff) const{
	LodMatrix mat(vec_size());
	// coef_u * u
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		mat.add_value(icell, icell, coef_u*_grid.cell_volume(icell));
	}
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);

		// - coef_diff * Laplace(u)
		for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
			size_t column = iter.first;
			double coef = -coef_diff * area * iter.second;
			mat.add_value(negative_colloc, column, coef);
			mat.add_value(positive_colloc, column, -coef);
		}
		
		// + coef_conv * convection
		{
			double coef = coef_conv * area * _un_face[iface]/2.0;
			mat.add_value(negative_colloc, negative_colloc,  coef);
			mat.add_value(negative_colloc, positive_colloc,  coef);
			mat.add_value(positive_colloc, positive_colloc, -coef);
			mat.add_value(positive_colloc, negative_colloc, -coef);
		}
	}
	// input, cylinder boundary: dirichlet
	for (size_t icolloc: _boundary_info.input) mat.set_unit_row(icolloc);
	for (size_t icolloc: _boundary_info.cyl) mat.set_unit_row(icolloc);
	// output boundary: du/dt + un*du/dn = 0
	for (size_t icolloc: _boundary_info.output){
		mat.remove_row(icolloc);
		size_t iface = _collocations.face_index(icolloc);
		double sgn = (_grid.tab_face_cell(iface)[1] == INVALID_INDEX) ? 1 : -1;
		mat.add_value(icolloc, icolloc, coef_u);
		for (auto it: _dfdn_computer.linear_combination(iface)){
			size_t icolumn = it.first;
			double value = coef_conv * sgn * it.second;
			mat.add_value(icolloc, icolumn, value);
		}
	}
	return mat.to_csr();
}

void CylinderFvmPimpleWorker::assemble_uv_slae(){
	// =============== LHS
	_mat_uv = assemble_uv_lhs(1+_tau/_time_step, _tau, _tau/_Re);
	_uv_solver.set_matrix(_mat_uv);

	// ============== RHS
	std::vector<Vector> grad_p = _grad_computer->compute(_p);
	_rhs_u.resize(vec_size());
	_rhs_v.resize(vec_size());
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		_rhs_u[icell] = (_u[icell] + _tau/_time_step*_u_old[icell] -_tau * grad_p[icell].x()) * _grid.cell_volume(icell);
		_rhs_v[icell] = (_v[icell] + _tau/_time_step*_v_old[icell] -_tau * grad_p[icell].y()) * _grid.cell_volume(icell);
	}
	// bnd
	for (size_t icolloc: _boundary_info.input){
		_rhs_u[icolloc] = 1;
		_rhs_v[icolloc] = 0;
	}
	for (size_t icolloc: _boundary_info.cyl){
		_rhs_u[icolloc] = 0;
		_rhs_v[icolloc] = 0;
	}
	for (size_t icolloc: _boundary_info.output){
		_rhs_u[icolloc] = _u[icolloc] + _u_old[icolloc] * _tau / _time_step;
		_rhs_v[icolloc] = _v[icolloc] + _v_old[icolloc] * _tau / _time_step;
	}
}

std::vector<double> CylinderFvmPimpleWorker::compute_u_star(){
	std::vector<double> u_star(_u);
	_uv_solver.solve(_rhs_u, u_star);
	return u_star;
}

std::vector<double> CylinderFvmPimpleWorker::compute_v_star(){
	std::vector<double> v_star(_v);
	_uv_solver.solve(_rhs_v, v_star);
	return v_star;
}

std::vector<double> CylinderFvmPimpleWorker::compute_un_face(
		const std::vector<double>& u_star,
		const std::vector<double>& v_star,
		double bval,
		bool rhie_chow, const std::vector<double>& p){

	std::vector<double> ret(_grid.n_faces());

	std::vector<Vector> grad_p;
	std::vector<double> dpdn_face;
	if (rhie_chow){
		grad_p = _grad_computer->compute(p);
		dpdn_face = _dfdn_computer.compute(p);
	}

	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		size_t ci = _grid.tab_face_cell(iface)[0];
		size_t cj = _grid.tab_face_cell(iface)[1];
		if (ci == INVALID_INDEX || cj == INVALID_INDEX){
			// Boundary. Default zero.
			ret[iface] = 0;
		} else {
			Vector normal = _grid.face_normal(iface);
			Vector uvec_i = Vector(u_star[ci], v_star[ci]);
			Vector uvec_j = Vector(u_star[cj], v_star[cj]);

			double ustar_i = dot_product(uvec_i, normal);
			double ustar_j = dot_product(uvec_j, normal);
			ret[iface] = 0.5 * (ustar_i + ustar_j);

			// Rhie-Chow correction
			if (rhie_chow){
				double dpdn_i = dot_product(grad_p[ci], normal);
				double dpdn_j = dot_product(grad_p[cj], normal);
				double dpdn_ij = dpdn_face[iface];
				ret[iface] += 0.5*_tau*(dpdn_i + dpdn_j - 2*dpdn_ij);
			}
		}
	}
	// input boundary 
	for (size_t icoll: _boundary_info.input){
		size_t iface = _collocations.face_index(icoll);
		ret[iface] = (_grid.tab_face_cell(iface)[0] == INVALID_INDEX) ? bval : -bval;
	}
	// output boundary
	for (size_t icoll: _boundary_info.output){
		size_t iface = _collocations.face_index(icoll);
		ret[iface] = (_grid.tab_face_cell(iface)[0] == INVALID_INDEX) ? -bval : bval;
	}

	return ret;
}

std::vector<double> CylinderFvmPimpleWorker::compute_un_face_prime(
		const std::vector<double>& p){
	std::vector<double> dpdn_face = _dfdn_computer.compute(p);
	std::vector<double> un(_grid.n_faces());
	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		un[iface] = -_tau * _d * dpdn_face[iface];
	}
	return un;
}

std::vector<double> CylinderFvmPimpleWorker::compute_p_prime(const std::vector<double>& un_star_face){
	// rhs
	std::vector<double> rhs(vec_size(), 0.0);
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		double coef = -_grid.face_area(iface) / _tau * un_star_face[iface];
		size_t neg = _grid.tab_face_cell(iface)[0];
		size_t pos = _grid.tab_face_cell(iface)[1];
		if (neg != INVALID_INDEX) rhs[neg] += coef;
		if (pos != INVALID_INDEX) rhs[pos] -= coef;
	}
	rhs[0] = 0;
	// solve
	std::vector<double> p_prime;
	_p_prime_solver.solve(rhs, p_prime);
	return p_prime;
}

std::vector<double> CylinderFvmPimpleWorker::compute_u_prime(const std::vector<Vector>& grad_p_prime){
	std::vector<double> u_prime(vec_size(), 0.0);
	for (size_t i=0; i<_grid.n_cells(); ++i){
		u_prime[i] = -_tau * _d * grad_p_prime[i].x();
	}
	return u_prime;
}

std::vector<double> CylinderFvmPimpleWorker::compute_v_prime(const std::vector<Vector>& grad_p_prime){
	std::vector<double> v_prime(vec_size(), 0.0);
	for (size_t i=0; i<_grid.n_cells(); ++i){
		v_prime[i] = -_tau * _d * grad_p_prime[i].y();
	}
	return v_prime;
}

double CylinderFvmPimpleWorker::to_next_time_step(){
	_u_old = _u;
	_v_old = _v;
	return to_next_iteration();
}

namespace {

std::string convergence_report(double time, int it){
	std::ostringstream oss;
	oss << std::setprecision(2) << std::fixed;
	oss << "Time = " << std::setw(5) << time << " converged in " << it << " iterations" << std::endl;
	return oss.str();
}

}

TEST_CASE("Cylinder 2D, FVM-PIMPLE algorithm", "[cylinder2-fvm-pimple]"){
	std::cout << std::endl << "--- cfd_test [cylinder2-fvm-pimple] --- " << std::endl;

	// problem parameters
	double Re = 100;
	size_t max_it = 10000;
	size_t n_piso = 0;
	double eps = 1e-1;
	double E = 4;
	double time_step = 0.5;
	double end_time = 0.5;

	// worker initialization
	auto grid = UnstructuredGrid2D::vtk_read(test_directory_file("cylgrid_5k.vtk"));
	CylinderFvmPimpleWorker worker(grid, Re, E, time_step, n_piso);
	worker.initialize_saver("cylinder2-fvm", 0.5);

	// time loop
	size_t it = 0;
	for (double time=time_step; time<end_time+1e-6; time+=time_step){
		for (it=1; it < max_it; ++it){
			double nrm = worker.step();
			// break inner iterations if residual is low enough
			if (nrm < eps){
				break;
			} else if (it == max_it-1) {
				std::cout << "WARNING: internal SIMPLE interations not converged with nrm = "
				          << nrm << std::endl;
			}
		}
		// uvp_old = uvp
		worker.to_next_time_step();

		// save and report
		std::cout << convergence_report(time, it);
		worker.save_current_fields(time);
	}

	CHECK(it == 13);
}
