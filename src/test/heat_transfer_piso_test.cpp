#include "cfd25_test.hpp"
#include "cfd/grid/vtk.hpp"
#include "cfd/grid/regular_grid2d.hpp"
#include "cfd/mat/lodmat.hpp"
#include "cfd/mat/sparse_matrix_solver.hpp"
#include "cfd/debug/printer.hpp"
#include "utils/vecmat.hpp"
#include "cfd/debug/tictoc.hpp"
#include <sstream>
#include <iomanip>

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

struct HeatTransferPisoWorker{
	HeatTransferPisoWorker(double Re, double Pe, const RegularGrid2D& grid, double time_step, size_t n_piso);

	void initialize();
	void step();

	void initialize_saver(bool save_exact_fields, std::string stem, double writer_step=0);
	void save_current_fields(double time);

	size_t u_size() const{
		return _yf_grid.n_points();
	}
	size_t v_size() const{
		return _xf_grid.n_points();
	}
	size_t p_size() const{
		return _cc_grid.n_points();
	}

	struct Coefficients{
		double cpx;
		double cfx;
		double cx;
	};
	Coefficients coefficients() const;
private:
	const RegularGrid2D _grid;
	const RegularGrid2D _cc_grid;
	const RegularGrid2D _xf_grid;
	const RegularGrid2D _yf_grid;
	const double _hx;
	const double _hy;
	const double _Re;
	const double _Pe;
	const double _time_step;
	const size_t _n_piso_steps;

	std::vector<double> _p;
	std::vector<double> _u;
	std::vector<double> _v;
	std::vector<double> _t;
	std::vector<double> _u_old;
	std::vector<double> _v_old;

	double _du;
	double _dv;
	AmgcMatrixSolver _p_prime_solver;

	CsrMatrix _mat_u;
	CsrMatrix _mat_v;
	std::vector<double> _rhs_u;
	std::vector<double> _rhs_v;

	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer_u;
	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer_v;
	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer_p;
	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer_all;
	std::ofstream cx_writer;

	void assemble_p_prime_solver();
	void assemble_u_slae();
	void assemble_v_slae();

	std::vector<double> compute_u_prime_outflow(const std::vector<double>& u_star) const;
	std::vector<double> compute_u_star();
	std::vector<double> compute_v_star();
	std::vector<double> compute_p_prime(
			const std::vector<double>& u_star,
			const std::vector<double>& v_star,
			const std::vector<double>& u_prime_outflow);
	std::vector<double> compute_u_prime(
			const std::vector<double>& p_prime,
			const std::vector<double>& u_prime_outflow,
			const std::vector<double>& addto={});
	std::vector<double> compute_v_prime(
			const std::vector<double>& p_prime,
			const std::vector<double>& addto={});
	std::vector<double> compute_temperature() const;
	double u_i_j(size_t i, size_t j) const;
	double u_ip_jp(size_t i, size_t j) const;
	double v_i_j(size_t i, size_t j) const;
	double v_ip_jp(size_t i, size_t j) const;
	double p_ip_jp(size_t i, size_t j) const;
	std::vector<Vector> build_main_grid_velocity() const;

	void to_next_time_step();
	void set_uvp(const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& p);
};

HeatTransferPisoWorker::HeatTransferPisoWorker(double Re, double Pe, const RegularGrid2D& grid, double time_step, size_t n_piso):
	_grid(grid),
	_cc_grid(_grid.cell_centered_grid()),
	_xf_grid(_grid.xface_centered_grid()),
	_yf_grid(_grid.yface_centered_grid()),
	_hx(_grid.Lx()/_grid.nx()),
	_hy(_grid.Ly()/_grid.ny()),
	_Re(Re),
	_Pe(Pe),
	_time_step(time_step),
	_n_piso_steps(n_piso)
{
	_du = 1.0 / (1 + 2.0*_time_step/_Re * (1.0/_hx/_hx + 1.0/_hy/_hy));
	_dv = 1.0 / (1 + 2.0*_time_step/_Re * (1.0/_hx/_hx + 1.0/_hy/_hy));
	assemble_p_prime_solver();
}

void HeatTransferPisoWorker::initialize(){
	_u = std::vector<double>(u_size(), 1);
	_v = std::vector<double>(v_size(), 0);
	_p = std::vector<double>(p_size(), 0);
	_t = std::vector<double>(p_size(), 0);

	to_next_time_step();
};


void HeatTransferPisoWorker::initialize_saver(bool save_exact_fields, std::string stem, double step){
	_writer_all.reset(new VtkUtils::TimeSeriesWriter(stem));
	if (save_exact_fields){
		_writer_u.reset(new VtkUtils::TimeSeriesWriter(stem + "-u"));
		_writer_v.reset(new VtkUtils::TimeSeriesWriter(stem + "-v"));
		_writer_p.reset(new VtkUtils::TimeSeriesWriter(stem + "-p"));
	}
	if (step > 0){
		_writer_all->set_time_step(step);
		if (save_exact_fields){
			_writer_u->set_time_step(step);
			_writer_v->set_time_step(step);
			_writer_p->set_time_step(step);
		}
	}
	cx_writer.open("c.txt");
	cx_writer << std::setw(12) << "Time"
	          << std::setw(12) << "Cpx"
	          << std::setw(12) << "Cfx"
	          << std::setw(12) << "Cx" << std::endl;

};

void HeatTransferPisoWorker::set_uvp(const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& p){
	_u = u;
	_v = v;
	_p = p;
	assemble_u_slae();
	assemble_v_slae();
};

void HeatTransferPisoWorker::to_next_time_step(){
	_t = compute_temperature();
	_u_old = _u;
	_v_old = _v;
	set_uvp(_u, _v, _p);
}

void HeatTransferPisoWorker::step(){
	// =========== SIMPLE step
	// Predictor step: U-star
	std::vector<double> u_star = compute_u_star();
	std::vector<double> v_star = compute_v_star();
	std::vector<double> u_prime_outflow = compute_u_prime_outflow(u_star);
	// Pressure correction
	std::vector<double> p_prime = compute_p_prime(u_star, v_star, u_prime_outflow);
	// Velocity correction
	std::vector<double> u_prime = compute_u_prime(p_prime, u_prime_outflow);
	std::vector<double> v_prime = compute_v_prime(p_prime);

	// =========== PISO steps
	std::vector<double> u_new = vector_sum(u_star, 1.0, u_prime);
	std::vector<double> v_new = vector_sum(v_star, 1.0, v_prime);
	std::vector<double> p_new = vector_sum(_p, 1.0, p_prime);
	std::fill(u_prime_outflow.begin(), u_prime_outflow.end(), 0.0);
	for (size_t k=0; k < _n_piso_steps; ++k){
		// -H*du*u'
		std::vector<double> hu = nodiag_mult(-_du, _mat_u, u_prime);
		// -H*dv*v'
		std::vector<double> hv = nodiag_mult(-_dv, _mat_v, v_prime);
		// pressure piso correction
		p_prime = compute_p_prime(hu, hv, u_prime_outflow);
		// velocity piso correction
		u_prime = compute_u_prime(p_prime, u_prime_outflow, hu);
		v_prime = compute_v_prime(p_prime, hv);

		for (size_t i=0; i<u_new.size(); ++i) u_new[i] += u_prime[i];
		for (size_t i=0; i<v_new.size(); ++i) v_new[i] += v_prime[i];
		for (size_t i=0; i<p_new.size(); ++i) p_new[i] += p_prime[i];
	}
	std::swap(_u, u_new);
	std::swap(_v, v_new);
	std::swap(_p, p_new);
	to_next_time_step();
}

void HeatTransferPisoWorker::save_current_fields(double time){
	std::vector<double> actnum(_grid.actnum().begin(), _grid.actnum().end());

	// all data on the main grid
	if (_writer_all){
		std::string filepath = _writer_all->add(time);
		if (filepath.size() > 0){
			_grid.save_vtk(filepath);
			VtkUtils::add_cell_data(_p, "pressure", filepath);
			VtkUtils::add_cell_data(_t, "temperature", filepath);
			VtkUtils::add_cell_data(actnum, "actnum", filepath);
			VtkUtils::add_point_vector(build_main_grid_velocity(), "velocity", filepath);
		}
	}
	// pressure
	if (_writer_p){
		std::string filepath = _writer_p->add(time);
		if (filepath.size() > 0){
			_cc_grid.save_vtk(filepath);
			VtkUtils::add_point_data(_p, "pressure", filepath);
			VtkUtils::add_point_data(_t, "temparature", filepath);
			VtkUtils::add_point_data(actnum, "actnum", filepath);
		}
	}
	// u
	if (_writer_u){
		std::string filepath = _writer_u->add(time);
		if (filepath.size() > 0){
			_yf_grid.save_vtk(filepath);
			VtkUtils::add_point_data(_u, "velocity-x", filepath);
		}
	}
	// v
	if (_writer_v){
		std::string filepath = _writer_v->add(time);
		if (filepath.size() > 0){
			_xf_grid.save_vtk(filepath);
			VtkUtils::add_point_data(_v, "velocity-y", filepath);
		}
	}
	
	// write coefficients to file
	Coefficients coefs = coefficients();
	cx_writer << std::setw(12) << time
		<< std::setw(12) << coefs.cpx
		<< std::setw(12) << coefs.cfx
		<< std::setw(12) << coefs.cx
		<< std::endl;
}


void HeatTransferPisoWorker::assemble_p_prime_solver(){
	LodMatrix mat(p_size());
	double coef_x = _du/_hx/_hx;
	double coef_y = _dv/_hy/_hy;
	for (size_t j = 0; j < _cc_grid.ny()+1; ++j)
	for (size_t i = 0; i < _cc_grid.nx()+1; ++i){
		size_t ind0 = _grid.cell_centered_grid_index_ip_jp(i, j);
		if (_grid.is_active_cell(ind0)){
			bool is_left = (i==0);
			bool is_right = (i==_cc_grid.nx());
			bool is_bottom = (j==0);
			bool is_top = (j==_cc_grid.ny());
			// x
			if (!is_right){
				size_t ind1 = _grid.cell_centered_grid_index_ip_jp(i+1, j);
				if (_grid.is_active_cell(ind1)){
					mat.add_value(ind0, ind0, coef_x);
					mat.add_value(ind0, ind1, -coef_x);
				}
			}
			if (!is_left){
				size_t ind1 = _grid.cell_centered_grid_index_ip_jp(i-1, j);
				if (_grid.is_active_cell(ind1)){
					mat.add_value(ind0, ind0, coef_x);
					mat.add_value(ind0, ind1, -coef_x);
				}
			}
			// y
			if (!is_top){
				size_t ind1 = _grid.cell_centered_grid_index_ip_jp(i, j+1);
				if (_grid.is_active_cell(ind1)){
					mat.add_value(ind0, ind0, coef_y);
					mat.add_value(ind0, ind1, -coef_y);
				}
			}
			if (!is_bottom){
				size_t ind1 = _grid.cell_centered_grid_index_ip_jp(i, j-1);
				if (_grid.is_active_cell(ind1)){
					mat.add_value(ind0, ind0, coef_y);
					mat.add_value(ind0, ind1, -coef_y);
				}
			}
		} else {
			mat.set_value(ind0, ind0, 1.0);
		}
	}
	mat.set_unit_row(0);
	_p_prime_solver.set_matrix(mat.to_csr());
}

void HeatTransferPisoWorker::assemble_u_slae(){
	_rhs_u.resize(_u.size());
	std::fill(_rhs_u.begin(), _rhs_u.end(), 0.0);
	LodMatrix mat(_u.size());

	auto add_to_mat = [&](size_t row_index, std::array<size_t, 2> ij_col, double value){
		if (ij_col[1] == _grid.ny()){
			// ghost index => top boundary condition: du/dn = 0
			size_t ind1 = _grid.yface_grid_index_i_jp(ij_col[0], ij_col[1]-1);
			mat.add_value(row_index, ind1, value);
		} else if (ij_col[1] == (size_t)-1){
			// ghost index => bottom boundary condition: du/dn = 0
			size_t ind1 = _grid.yface_grid_index_i_jp(ij_col[0], ij_col[1]+1);
			mat.add_value(row_index, ind1, value);
		} else {
			size_t ind1 = _grid.yface_grid_index_i_jp(ij_col[0], ij_col[1]);
			if (_grid.yface_type(ind1) == RegularGrid2D::FaceType::Deactivated){
				// ghost index => obstacle boundary u = 0
				mat.add_value(row_index, row_index, -value);
			} else {
				mat.add_value(row_index, ind1, value);
			}
		}
	};

	for (size_t j=0; j< _grid.ny(); ++j){
		// left boundary: u = 1
		{
			size_t index_left = _grid.yface_grid_index_i_jp(0, j);
			add_to_mat(index_left, {0, j}, 1.0);
			_rhs_u[index_left] = 1.0;
		}
		// right boundary: du/dt + u*du/dx = 0
		{
			size_t index_right = _grid.yface_grid_index_i_jp(_grid.nx(), j);
			size_t index_right_prev = _grid.yface_grid_index_i_jp(_grid.nx()-1, j);
			double u0 = std::max(0.0, _u[index_right]);
			double coef = _time_step*u0/_hx;
			mat.set_value(index_right, index_right, 1.0 + coef);
			mat.set_value(index_right, index_right_prev, -coef);
			_rhs_u[index_right] = u0 + _time_step * _u_old[index_right];
		}
	}

	// internal
	for (size_t j=0; j < _grid.ny(); ++j)
	for (size_t i=1; i < _grid.nx(); ++i){
		size_t row_index = _grid.yface_grid_index_i_jp(i, j);   //[i, j+1/2] linear index in u grid
		if (_grid.yface_type(row_index) == RegularGrid2D::FaceType::Internal){
			double u0_plus   = u_ip_jp(i, j);   // _u[i+1/2, j+1/2]
			double u0_minus  = u_ip_jp(i-1, j); // _u[i-1/2, j+1/2]
			double v0_plus   = v_i_j(i, j+1);   // _v[i,j+1]
			double v0_minus  = v_i_j(i, j);     // _v[i,j]

			// u_(i,j+1/2)
			add_to_mat(row_index, {i, j}, 1.0);
			//     + time_step * d(u0*u)/ dx
			add_to_mat(row_index, {i+1,j}, _time_step/2.0/_hx*u0_plus);
			add_to_mat(row_index, {i-1,j}, -_time_step/2.0/_hx*u0_minus);
			//     + tau * d(v0*u)/dy
			add_to_mat(row_index, {i, j+1}, _time_step/2.0/_hy*v0_plus);
			add_to_mat(row_index, {i, j-1}, -_time_step/2.0/_hy*v0_minus);
			//     - tau / Re * d^2u/dx^2
			add_to_mat(row_index, {i, j}, 2.0*_time_step/_Re/_hx/_hx);
			add_to_mat(row_index, {i+1, j}, -_time_step/_Re/_hx/_hx);
			add_to_mat(row_index, {i-1, j}, -_time_step/_Re/_hx/_hx);
			//     - tau / Re * d^2u/dy^2
			add_to_mat(row_index, {i, j}, 2.0*_time_step/_Re/_hy/_hy);
			add_to_mat(row_index, {i, j+1}, -_time_step/_Re/_hy/_hy);
			add_to_mat(row_index, {i, j-1}, -_time_step/_Re/_hy/_hy);
			// = u_old(i, j+1/2)
			_rhs_u[row_index] += _u_old[row_index];
			//      - tau * dp/dx
			_rhs_u[row_index] -= _time_step/_hx*(p_ip_jp(i, j) - p_ip_jp(i-1, j));
		} else {
			mat.set_value(row_index, row_index, 1.0);
			_rhs_u[row_index] = 0;
		}
	}
	_mat_u = mat.to_csr();
}

void HeatTransferPisoWorker::assemble_v_slae(){
	_rhs_v.resize(_v.size());
	std::fill(_rhs_v.begin(), _rhs_v.end(), 0.0);
	LodMatrix mat(_v.size());

	auto add_to_mat = [&](size_t row_index, std::array<size_t, 2> ij_col, double value){
		if (ij_col[0] == (size_t)-1){
			// left boundary condition: v = 0
			size_t ind1 = _grid.xface_grid_index_ip_j(ij_col[0]+1, ij_col[1]);
			mat.add_value(row_index, ind1, -value);
		} else if (ij_col[0] == _grid.nx()){
			// right boundary condition: v = 0
			size_t ind1 = _grid.xface_grid_index_ip_j(ij_col[0]-1, ij_col[1]);
			mat.add_value(row_index, ind1, -value);
		} else {
			// internal
			size_t ind1 = _grid.xface_grid_index_ip_j(ij_col[0], ij_col[1]);
			if (_grid.xface_type(ind1) == RegularGrid2D::FaceType::Deactivated){
				mat.add_value(row_index, row_index, -value);
			} else {
				mat.add_value(row_index, ind1, value);
			}
		}
	};

	// === top/bottom boundaries 
	for (size_t i=0; i<_grid.nx(); ++i){
		// top boundary condition: v = 0;
		size_t index_top = _grid.xface_grid_index_ip_j(i, _grid.ny());
		add_to_mat(index_top, {i, _grid.ny()}, 1.0);
		_rhs_v[index_top] = 0.0;
		// bottom boundary condition: v = 0;
		size_t index_bottom = _grid.xface_grid_index_ip_j(i, 0);
		add_to_mat(index_bottom, {i, 0}, 1.0);
		_rhs_v[index_bottom] = 0.0;
	}

	// === internal
	for (size_t j=1; j<_grid.ny(); ++j)
	for (size_t i=0; i<_grid.nx(); ++i){
		size_t row_index = _grid.xface_grid_index_ip_j(i, j);   //[i+1/2, j] linear index in v grid
	
		if (_grid.xface_type(row_index) == RegularGrid2D::FaceType::Internal){
			double u0_plus   = u_i_j(i+1, j);   // _u[i+1, j]
			double u0_minus  = u_i_j(i, j);     // _u[i, j]
			double v0_plus   = v_ip_jp(i, j);   // _v[i+1/2, j+1/2]
			double v0_minus  = v_ip_jp(i, j-1); // _v[i+1/2, j-1/2]

			// v_(i+1/2, j)
			add_to_mat(row_index, {i, j}, 1.0);
			//     + tau * d (u0*v) / dx
			add_to_mat(row_index, {i+1, j}, _time_step/2.0/_hx*u0_plus);
			add_to_mat(row_index, {i-1, j}, -_time_step/2.0/_hx*u0_minus);
			//     + tau * d (v0*v) / dy
			add_to_mat(row_index, {i, j+1}, _time_step/2.0/_hy*v0_plus);
			add_to_mat(row_index, {i, j-1}, -_time_step/2.0/_hy*v0_minus);
			//     - tau/Re * d^2v/dx^2
			add_to_mat(row_index, {i, j}, 2.0*_time_step/_Re/_hx/_hx);
			add_to_mat(row_index, {i+1, j}, -_time_step/_Re/_hx/_hx);
			add_to_mat(row_index, {i-1, j}, -_time_step/_Re/_hx/_hx);
			//     - tau/Re * d^2v/dy^2
			add_to_mat(row_index, {i, j}, 2.0*_time_step/_Re/_hy/_hy);
			add_to_mat(row_index, {i, j+1}, -_time_step/_Re/_hy/_hy);
			add_to_mat(row_index, {i, j-1}, -_time_step/_Re/_hy/_hy);
			// =  v_old
			_rhs_v[row_index] += _v_old[row_index];
			//    - tau * dp/dy
			_rhs_v[row_index] -= _time_step*(p_ip_jp(i, j) - p_ip_jp(i, j-1))/_hy;
		} else {
			// v = 0 on boundary and deactivated faces
			mat.set_value(row_index, row_index, 1.0);
			_rhs_v[row_index] = 0;
		}
	}
	_mat_v = mat.to_csr();
}


std::vector<double> HeatTransferPisoWorker::compute_u_star(){
	std::vector<double> u_star(_u);
	AmgcMatrixSolver::solve_slae(_mat_u, _rhs_u, u_star);
	return u_star;
}

std::vector<double> HeatTransferPisoWorker::compute_v_star(){
	std::vector<double> v_star(_v);
	AmgcMatrixSolver::solve_slae(_mat_v, _rhs_v, v_star);
	return v_star;
}

std::vector<double> HeatTransferPisoWorker::compute_p_prime(
		const std::vector<double>& u_star,
		const std::vector<double>& v_star,
		const std::vector<double>& u_prime_outflow){
	std::vector<double> rhs(_p.size(), 0.0);
	for (size_t i = 0; i < _grid.nx(); ++i)
	for (size_t j = 0; j < _grid.ny(); ++j){
		size_t ind0 = _grid.cell_centered_grid_index_ip_jp(i, j);
		size_t ind_left = _grid.yface_grid_index_i_jp(i, j);
		size_t ind_right = _grid.yface_grid_index_i_jp(i+1, j);
		size_t ind_bot = _grid.xface_grid_index_ip_j(i, j);
		size_t ind_top = _grid.xface_grid_index_ip_j(i, j+1);
		rhs[ind0] = -(u_star[ind_right] - u_star[ind_left])/_time_step/_hx - (v_star[ind_top] - v_star[ind_bot])/_time_step/_hy;
		// outflow compensation
		if (i == _grid.nx()-1){
			rhs[ind0] -= (u_prime_outflow[j]) / _time_step / _hx;
		}
	}
	rhs[0] = 0;
	std::vector<double> p_prime;
	_p_prime_solver.solve(rhs, p_prime);
	double p0 = (p_prime[0] + p_prime[_grid.cell_centered_grid_index_ip_jp(0, _grid.ny()-1)])/2.0;
	for (size_t i=0; i<p_prime.size(); ++i){
		if (_grid.is_active_cell(i)){
			p_prime[i] -= p0;
		} else {
			p_prime[i] = 0;
		}
	}
	return p_prime;
}

std::vector<double> HeatTransferPisoWorker::compute_u_prime_outflow(const std::vector<double>& u_star) const{
	double qin = 0;
	double qout = 0;
	for (size_t j=0; j<_grid.ny(); ++j){
		size_t ind_left = _grid.yface_grid_index_i_jp(0, j);
		size_t ind_right = _grid.yface_grid_index_i_jp(_grid.nx(), j);
		qin += u_star[ind_left]*_hy;
		qout += u_star[ind_right]*_hy;
	}
	double Lout = _grid.Ly();
	double diff_u = (qin - qout)/Lout;
	std::vector<double> ret(_grid.ny(), diff_u);
	return ret;
}

std::vector<double> HeatTransferPisoWorker::compute_u_prime(
		const std::vector<double>& p_prime,
		const std::vector<double>& u_prime_outflow,
		const std::vector<double>& addto){
	std::vector<double> u_prime = (addto.empty()) ? std::vector<double>(_u.size(), 0.0) : addto;
	for (size_t i=1; i<_grid.nx(); ++i)
	for (size_t j=0; j<_grid.ny(); ++j){
		size_t ind0 = _grid.yface_grid_index_i_jp(i, j);
		if (_grid.yface_type(ind0) == RegularGrid2D::FaceType::Internal){
			size_t ind_plus  = _grid.cell_centered_grid_index_ip_jp(i, j);
			size_t ind_minus = _grid.cell_centered_grid_index_ip_jp(i-1, j);
			u_prime[ind0] = -_time_step * _du * (p_prime[ind_plus] - p_prime[ind_minus])/_hx;
		}
	}
	// outflow
	for (size_t j=0; j<_grid.ny(); ++j){
		size_t ind0 = _grid.yface_grid_index_i_jp(_grid.nx(), j);
		u_prime[ind0] = u_prime_outflow[j];
	}
	return u_prime;
}

std::vector<double> HeatTransferPisoWorker::compute_v_prime(
		const std::vector<double>& p_prime,
		const std::vector<double>& addto){
	std::vector<double> v_prime = (addto.empty()) ? std::vector<double>(_v.size(), 0.0) : addto;
	for (size_t i=0; i<_grid.nx(); ++i)
	for (size_t j=1; j<_grid.ny(); ++j){
		size_t ind0 = _grid.xface_grid_index_ip_j(i, j);
		if (_grid.xface_type(ind0) == RegularGrid2D::FaceType::Internal){
			size_t ind_plus  = _grid.cell_centered_grid_index_ip_jp(i, j);
			size_t ind_minus = _grid.cell_centered_grid_index_ip_jp(i, j-1);
			v_prime[ind0] = -_time_step * _dv * (p_prime[ind_plus] - p_prime[ind_minus])/_hy;
		}
	}
	return v_prime;
}

std::vector<double> HeatTransferPisoWorker::compute_temperature() const{
	std::vector<double> rhs(_p.size(), 0);
	LodMatrix mat(_p.size());
	auto add_to_mat = [&](size_t row_index, std::array<size_t, 2> ij_col, double value){
		if (ij_col[0] == (size_t)-1){
			// left boundary: T=0
			mat.add_value(row_index, row_index, -value);
		} else if (ij_col[0] == _grid.nx()){
			// right boundary: dT/dn = 0
			mat.add_value(row_index, row_index, value);
		} else if (ij_col[1] == (size_t)-1){
			// bottom boundary: dT/dn = 0
			mat.add_value(row_index, row_index, value);
		} else if (ij_col[1] == _grid.ny()){
			// bottom boundary: dT/dn = 0
			mat.add_value(row_index, row_index, value);
		} else {
			size_t ind1 = _grid.cell_centered_grid_index_ip_jp(ij_col[0], ij_col[1]);
			if (_grid.is_active_cell(ind1)){
				mat.add_value(row_index, ind1, value);
			} else {
				// internal boundary: T = 1
				mat.add_value(row_index, row_index, -value);
				rhs[row_index] -= 2*value;
			}
		}
	};
	for (size_t j=0; j < _grid.ny(); ++j)
	for (size_t i=0; i < _grid.nx(); ++i){
		size_t row_index = _grid.cell_centered_grid_index_ip_jp(i, j);
		if (_grid.is_active_cell(row_index)){
			double u_left = _u[_grid.yface_grid_index_i_jp(i, j)];
			double u_right = _u[_grid.yface_grid_index_i_jp(i+1, j)];
			double v_bot = _v[_grid.xface_grid_index_ip_j(i, j)];
			double v_top = _v[_grid.xface_grid_index_ip_j(i, j+1)];

			// 1.0/time_step T(i+1/2,j+1/2)
			add_to_mat(row_index, {i, j}, 1.0 / _time_step);
			//     + d(u0*T)/ dx
			add_to_mat(row_index, {i+1,j},  u_right/2.0/_hx);
			add_to_mat(row_index, {i-1,j}, -u_left/2.0/_hx);
			//     + d(v0*T)/dy
			add_to_mat(row_index, {i, j+1},  v_top/2.0/_hy);
			add_to_mat(row_index, {i, j-1}, -v_bot/2.0/_hy);
			//     - 1 / Re * d^2u/dx^2
			add_to_mat(row_index, {i, j},    2.0/_Pe/_hx/_hx);
			add_to_mat(row_index, {i+1, j}, -1.0/_Pe/_hx/_hx);
			add_to_mat(row_index, {i-1, j}, -1.0/_Pe/_hx/_hx);
			//     - 1 / Re * d^2u/dy^2
			add_to_mat(row_index, {i, j},     2.0/_Pe/_hy/_hy);
			add_to_mat(row_index, {i, j+1},  -1.0/_Pe/_hy/_hy);
			add_to_mat(row_index, {i, j-1},  -1.0/_Pe/_hy/_hy);
			// = 1.0 / time_step*Told
			rhs[row_index] += 1.0 / _time_step*_t[row_index];
		} else {
			mat.set_value(row_index, row_index, 1.0);
			rhs[row_index] = 0;
		}
	}
	std::vector<double> temperature;
	AmgcMatrixSolver::solve_slae(mat.to_csr(), rhs, temperature);
	return temperature;
}

double HeatTransferPisoWorker::u_i_j(size_t i, size_t j) const{
	size_t ind0 = _grid.yface_grid_index_i_jp(i, j);
	size_t ind1 = _grid.yface_grid_index_i_jp(i, j-1);
	return (_u[ind0] + _u[ind1])/2.0;
}
double HeatTransferPisoWorker::u_ip_jp(size_t i, size_t j) const{
	size_t ind0 = _grid.yface_grid_index_i_jp(i, j);
	size_t ind1 = _grid.yface_grid_index_i_jp(i+1, j);
	return (_u[ind0] + _u[ind1])/2.0;
}

double HeatTransferPisoWorker::v_i_j(size_t i, size_t j) const{
	size_t ind0 = _grid.xface_grid_index_ip_j(i, j);
	size_t ind1 = _grid.xface_grid_index_ip_j(i-1, j);
	return (_v[ind0] + _v[ind1])/2.0;
}
double HeatTransferPisoWorker::v_ip_jp(size_t i, size_t j) const{
	size_t ind0 = _grid.xface_grid_index_ip_j(i, j);
	size_t ind1 = _grid.xface_grid_index_ip_j(i, j+1);
	return (_v[ind0] + _v[ind1])/2.0;
}

double HeatTransferPisoWorker::p_ip_jp(size_t i, size_t j) const{
	return _p[_grid.cell_centered_grid_index_ip_jp(i, j)];
}

std::vector<Vector> HeatTransferPisoWorker::build_main_grid_velocity() const{
	std::vector<Vector> ret(_grid.n_points());
	// boundary
	for (size_t i = 0; i < _grid.nx()+1; ++i){
		{
			// bottom boundary
			size_t ind_bot = _grid.to_linear_point_index({i, 0});
			double u = _u[_grid.yface_grid_index_i_jp(i, 0)];
			ret[ind_bot] = Vector(u, 0);
		}
		{
			// top boundary
			size_t ind_top = _grid.to_linear_point_index({i, _grid.ny()});
			double u = _u[_grid.yface_grid_index_i_jp(i, _grid.ny()-1)];
			ret[ind_top] = Vector(u, 0);
		}
	}
	for (size_t j = 0; j < _grid.ny()+1; ++j){
		// left boundary
		size_t ind_left = _grid.to_linear_point_index({0, j});
		ret[ind_left] = Vector(1, 0);
		// right boundary
		size_t ind_right = _grid.to_linear_point_index({_grid.nx(), j});
		double u_up, u_down;
		if (j < _grid.ny()){
			u_up = _u[_grid.yface_grid_index_i_jp(_grid.nx(), j)];
		} else {
			u_up = _u[_grid.yface_grid_index_i_jp(_grid.nx(), j-1)];
		}
		if (j > 0){
			u_down = _u[_grid.yface_grid_index_i_jp(_grid.nx(), j-1)];
		} else {
			u_down = _u[_grid.yface_grid_index_i_jp(_grid.nx(), j)];
		}
		double u = (u_up + u_down)/2.0;
		double v = _v[_grid.xface_grid_index_ip_j(_grid.nx()-1, j)];
		ret[ind_right] = Vector(u, v);
	}
	// internal
	for (size_t j=1; j<_grid.ny(); ++j)
	for (size_t i=1; i<_grid.nx(); ++i){
		size_t ind = _grid.to_linear_point_index({i, j});
		size_t ind_top = _grid.yface_grid_index_i_jp(i, j);
		size_t ind_bot = _grid.yface_grid_index_i_jp(i, j-1);
		size_t ind_left = _grid.xface_grid_index_ip_j(i-1, j);
		size_t ind_right = _grid.xface_grid_index_ip_j(i, j);
		double u, v;
		if (_grid.yface_type(ind_top) != RegularGrid2D::FaceType::Internal ||
		    _grid.yface_type(ind_bot) != RegularGrid2D::FaceType::Internal){
			u = 0.0;
		} else {
			u = 0.5*(_u[ind_top] + _u[ind_bot]);
		}
		if (_grid.xface_type(ind_left) != RegularGrid2D::FaceType::Internal ||
		    _grid.xface_type(ind_right) != RegularGrid2D::FaceType::Internal){
			v = 0.0;
		} else {
			v = 0.5*(_v[ind_left] + _v[ind_right]);
		}
		ret[ind] = Vector(u, v);
	}

	return ret;
}

HeatTransferPisoWorker::Coefficients HeatTransferPisoWorker::coefficients() const{
	double sum_cpx = 0;
	double sum_cfx = 0;

	// yfaces loop
	for (const RegularGrid2D::split_index_t& yface: _grid.boundary_yfaces()){
		if (yface[0] == 0){
			// input => ignore
		} else if (yface[0] == _grid.nx()){
			// output => ignore
		} else {
			double pnx;
			size_t left_cell = _grid.cell_centered_grid_index_ip_jp(yface[0]-1, yface[1]);
			size_t right_cell = _grid.cell_centered_grid_index_ip_jp(yface[0], yface[1]);
			if (_grid.is_active_cell(left_cell)){
				pnx = _p[left_cell];  
			} else if (_grid.is_active_cell(right_cell)){
				pnx = -_p[right_cell];  
			} else {
				_THROW_UNREACHABLE_;
			}
			sum_cpx += pnx * _hy;
		}
	}

	// xfaces loop
	for (const RegularGrid2D::split_index_t& xface: _grid.boundary_xfaces()){
		if (xface[1] == 0){
			// bottom => ignore
		} else if (xface[1] == _grid.ny()){
			// top => ignore
		} else {
			double dudn;
			size_t bot_cell = _grid.cell_centered_grid_index_ip_jp(xface[0], xface[1]-1);
			size_t top_cell = _grid.cell_centered_grid_index_ip_jp(xface[0], xface[1]);
			if (_grid.is_active_cell(bot_cell)){
				dudn = -u_ip_jp(xface[0], xface[1]-1)/(_hy/2.0);
			} else if (_grid.is_active_cell(top_cell)){
				dudn = -u_ip_jp(xface[0], xface[1])/(_hy/2.0);
			} else {
				_THROW_UNREACHABLE_;
			}
			sum_cfx += dudn * _hx;
		}
	}

	Coefficients coefs;
	coefs.cpx = 2.0*sum_cpx;
	coefs.cfx = -2.0/_Re*sum_cfx;
	coefs.cx = coefs.cpx + coefs.cfx;
	return coefs;
}

TEST_CASE("Heat transfer, PISO algorithm", "[heat-transfer-piso]"){
	std::cout << std::endl << "--- cfd_test [heat-transfer-piso] --- " << std::endl;

	// problem parameters
	double Re = 100;
	double Pe = 100;
	size_t n_unit = 10;  // partition per unit length
	double time_step = 0.05;
	double end_time = 1.0;
	size_t n_piso = 1;

	// worker initialization
	RegularGrid2D grid(0, 12, -2, 2, 12*n_unit, 4*n_unit);
	grid.deactivate_cells({2, -0.5}, {3, 0.5});
	HeatTransferPisoWorker worker(Re, Pe, grid, time_step, n_piso);
	worker.initialize_saver(false, "heat_transfer_piso", 0.5);

	// initial condition
	worker.initialize();
	worker.save_current_fields(0);

	// iterations loop
	for (double time=time_step; time<end_time+1e-6; time+=time_step){
		std::cout << "time = " << time << std::endl;
		worker.step();
		worker.save_current_fields(time);
	}
	CHECK(worker.coefficients().cx == Approx(2.55).margin(1e-2));
}
