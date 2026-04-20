#include "cfd/debug/printer.hpp"
#include "cfd/debug/saver.hpp"
#include "cfd/debug/tictoc.hpp"
#include "cfd/fvm/fvm_assembler.hpp"
#include "cfd/grid/regular_grid2d.hpp"
#include "cfd/grid/unstructured_grid2d.hpp"
#include "cfd/grid/vtk.hpp"
#include "cfd/mat/lodmat.hpp"
#include "cfd/mat/sparse_matrix_solver.hpp"
#include "cfd25_test.hpp"
#include "utils/filesystem.hpp"
#include "utils/vecmat.hpp"
#include <iomanip>
#include <list>

using namespace cfd;

namespace {

constexpr double ALPHA_P = 0.3;
constexpr double ALPHA_U = 0.8;

constexpr double ALPHA_K = 0.5;
constexpr double ALPHA_E = 0.5;

constexpr double C_MU = 0.09;
constexpr double SIGMA_K = 1.0;
constexpr double SIGMA_E = 1.3;
constexpr double C1 = 1.44;
constexpr double C2 = 1.92;
constexpr double DELTA_NUT = 0.01;   // nuT >= DELTA_NUT * 1/Re

double Limiter(double r){
	//return 0.0; // UPWIND
	//return 1.0; // Symmetric
	return std::min(1.0, std::max(0.0, r));   // MINMOD
	//return std::max(0.0, std::min(std::min(2.0*r, (1+r)/2.0), 2.0));  // MC
}

double tvd_upwind_weight(
	const std::vector<double>& u,
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
	double phi = Limiter(r);
	return 1 - phi/2;
}

void relax_slae(CsrMatrix& mat, std::vector<double>& rhs, const std::vector<double>& uold, double alpha, size_t n_rows){
	for (size_t irow=0; irow<n_rows; ++irow){
		double& a_ii = mat.vals()[mat.get_address(irow, irow)];
		rhs[irow] += (1 - alpha) / alpha * a_ii * uold[irow];
		a_ii /= alpha;
	}
};

struct CavityFvmWorker {
	CavityFvmWorker(const IGrid& grid, double Re);
	void initialize_saver(std::string stem, size_t save_iter_step);

	double step();
	void save_current_fields(size_t iter) const;
	void to_next_time_step() {};

	size_t vec_size() const {
		return collocations_.size();
	}

private:
	const IGrid& grid_;
	const double Re_;
	const FvmExtendedCollocations collocations_;
	const FvmFacesDn dfdn_computer_;
	const std::shared_ptr<IFvmCellGradient> grad_computer_;

	struct BoundaryInfo {
		std::vector<size_t> all;
		std::vector<size_t> top;
		std::vector<size_t> nontop;
		std::map<size_t, std::vector<size_t>> bnd_cell_to_colloc; // near boundary cell -> boundary collocations
	};
	BoundaryInfo boundary_info_;

	std::vector<double> p_;
	std::vector<double> u_;
	std::vector<double> v_;
	std::vector<double> du_;
	std::vector<double> dv_;
	std::vector<double> un_face_;
	std::vector<double> dn_face_;
	std::vector<double> dpdn_face_;
	std::vector<Vector> grad_p_;
	std::vector<double> k_;
	std::vector<double> epsilon_;
	std::vector<double> nuT_;


	CsrMatrix mat_u_;
	CsrMatrix mat_v_;
	std::vector<double> rhs_u_;
	std::vector<double> rhs_v_;

	std::shared_ptr<VtkUtils::TimeSeriesWriter> writer_;

	CsrMatrix assemble_convdiff_lhs(
		const std::vector<double>& u,
		const std::vector<double>& nu_face,
		const std::vector<double>& diag={}) const;
	void prepare_velocity_slae();
	std::vector<double> compute_u_tilde(const CsrMatrix& A, const std::vector<double>& d,
	                                    const std::vector<double>& u_star, const std::vector<double>& u) const;
	std::vector<double> compute_p(const std::vector<double>& un_face) const;
	std::vector<double> compute_un_face(const std::vector<double>& x, const std::vector<double>& y) const;
	std::vector<double> compute_dn_face(const std::vector<double>& x, const std::vector<double>& y) const;
	std::vector<double> compute_un_star_face_rhie_chow(const std::vector<double>& u_star,
                                                           const std::vector<double>& v_star) const;
	// k-e specific
	std::vector<double> compute_nueff_face(double sigma=1.0) const;
	std::vector<double> compute_G() const;
	std::vector<double> compute_gamma() const;
	std::vector<double> compute_nuT() const;
	std::vector<double> compute_epsilon(const std::vector<double>& G, const std::vector<double>& gamma) const;
	std::vector<double> compute_k(const std::vector<double>& G, const std::vector<double>& gamma) const;


	double to_next_iteration();
	void gather_boundary_collocations();
};

CavityFvmWorker::CavityFvmWorker(const IGrid& grid, double Re)
	: grid_(grid),
	  Re_(Re),
	  collocations_(grid),
	  dfdn_computer_(grid, collocations_),
	  grad_computer_(std::make_shared<GaussLinearFvmCellGradient>(grid, collocations_)) {

	gather_boundary_collocations();

	rhs_u_ = std::vector<double>(vec_size(), 0);
	rhs_v_ = std::vector<double>(vec_size(), 0);
	du_ = std::vector<double>(vec_size(), 0);
	dv_ = std::vector<double>(vec_size(), 0);
	u_ = std::vector<double>(vec_size(), 0);
	v_ = std::vector<double>(vec_size(), 0);
	p_ = std::vector<double>(vec_size(), 0);
	un_face_ = std::vector<double>(grid_.n_faces(), 0);
	dn_face_ = std::vector<double>(grid_.n_faces(), 0);

	dpdn_face_ = std::vector<double>(grid_.n_faces(), 0);
	grad_p_ = std::vector<Vector>(grid_.n_cells(), Vector{0, 0, 0});

	k_ = std::vector<double>(vec_size(), 0.0);
	epsilon_ = std::vector<double>(vec_size(), 0.0);
	nuT_ = std::vector<double>(vec_size(), DELTA_NUT / Re_);

	to_next_iteration();
}

void CavityFvmWorker::gather_boundary_collocations() {
	BoundaryInfo& bi = boundary_info_;

	for (size_t icolloc: collocations_.face_collocations){
		size_t iface = collocations_.face_index(icolloc);
		bi.all.push_back(icolloc);
		if (std::abs(grid_.face_center(iface).y() - 1) < 1e-6){
			bi.top.push_back(icolloc);
		} else {
			bi.nontop.push_back(icolloc);
		}
	}

	for (size_t icolloc: bi.all){
		size_t iface = collocations_.face_index(icolloc);
		size_t icell = grid_.tab_face_cell(iface)[0];
		if (icell == INVALID_INDEX) {
			icell = grid_.tab_face_cell(iface)[1];
		}
		auto fnd = bi.bnd_cell_to_colloc.find(icell);
		if (fnd == bi.bnd_cell_to_colloc.end()){
			bi.bnd_cell_to_colloc[icell] = {};
		}
		bi.bnd_cell_to_colloc[icell].push_back(icolloc);
	}
}

void CavityFvmWorker::initialize_saver(std::string stem, size_t save_iter_step) {
	writer_.reset(new VtkUtils::TimeSeriesWriter(stem));
	writer_->set_time_step(save_iter_step);
};

double CavityFvmWorker::to_next_iteration() {
	prepare_velocity_slae();

	// residual vectors
	std::vector<double> res_u = compute_residual_vec(mat_u_, rhs_u_, u_);
	std::vector<double> res_v = compute_residual_vec(mat_v_, rhs_v_, v_);
	// norm
	double res = 0;
	for (size_t icell = 0; icell < grid_.n_cells(); ++icell) {
		res = std::max(res, std::max(res_u[icell], res_v[icell]) / grid_.cell_volume(icell));
	}
	return res;
};

std::vector<double> CavityFvmWorker::compute_un_star_face_rhie_chow(const std::vector<double>& u_star,
                                                                    const std::vector<double>& v_star) const {

	std::vector<double> ret(grid_.n_faces());

	for (size_t iface = 0; iface < grid_.n_faces(); ++iface) {
		size_t ci = grid_.tab_face_cell(iface)[0];
		size_t cj = grid_.tab_face_cell(iface)[1];
		if (ci == INVALID_INDEX || cj == INVALID_INDEX) {
			ret[iface] = 0;
		} else {
			// Rhie-Chow interpolation
			Vector normal = grid_.face_normal(iface);
			Vector uvec_i = Vector(u_star[ci], v_star[ci]);
			Vector uvec_j = Vector(u_star[cj], v_star[cj]);

			double ustar_i = dot_product(uvec_i, normal);
			double ustar_j = dot_product(uvec_j, normal);
			double d_dpdn_i = dot_product(Vector{grad_p_[ci].x() * du_[ci], grad_p_[ci].y() * dv_[ci]}, normal);
			double d_dpdn_j = dot_product(Vector{grad_p_[cj].x() * du_[cj], grad_p_[cj].y() * dv_[cj]}, normal);
			double d_dpdn_ij = dn_face_[iface] * dpdn_face_[iface];

			ret[iface] = 0.5 * (ustar_i + ustar_j) + 0.5 * (d_dpdn_i + d_dpdn_j - 2 * d_dpdn_ij);
		}
	}

	return ret;
}

double CavityFvmWorker::step() {
	// 1. Predictor step: U-star
	std::vector<double> u_star = u_, v_star = v_;
	AmgcMatrixSolver::solve_slae(mat_u_, rhs_u_, u_star);
	AmgcMatrixSolver::solve_slae(mat_v_, rhs_v_, v_star);
	std::vector<double> un_star_face = compute_un_star_face_rhie_chow(u_star, v_star);
	// 2. Pressure correction
	std::vector<double> p_prime = compute_p(un_star_face);
	std::vector<Vector> grad_p_prime = grad_computer_->compute(p_prime);
	std::vector<double> dpprime_dn_face = dfdn_computer_.compute(p_prime);
	// 3. Velocity correction
	std::vector<double> u_prime(vec_size(), 0.0), v_prime(vec_size(), 0.0), un_prime_face(grid_.n_faces());
	for (size_t i = 0; i < grid_.n_cells(); ++i) {
		u_prime[i] = -du_[i] * grad_p_prime[i].x();
		v_prime[i] = -dv_[i] * grad_p_prime[i].y();
	}
	for (size_t iface = 0; iface < grid_.n_faces(); ++iface) {
		un_prime_face[iface] = -dn_face_[iface] * dpprime_dn_face[iface];
	}
	// 4. Set final values
	u_ = vector_sum(u_star, 1.0, u_prime);
	v_ = vector_sum(v_star, 1.0, v_prime);
	un_face_ = vector_sum(un_star_face, 1.0, un_prime_face);
	p_ = vector_sum(p_, ALPHA_P, p_prime);
	grad_p_ = vector_sum(grad_p_, ALPHA_P, grad_p_prime);
	dpdn_face_ = vector_sum(dpdn_face_, ALPHA_P, dpprime_dn_face);

	// Generation
	std::vector<double> G = compute_G();
	// gamma
	std::vector<double> gamma = compute_gamma();
	// epsilon
	epsilon_ = compute_epsilon(G, gamma);
	// k
	k_ = compute_k(G, gamma);
	// turbulent viscosity
	nuT_ = compute_nuT();

	return to_next_iteration();
}

void CavityFvmWorker::save_current_fields(size_t iter) const {
	if (writer_) {
		std::string filepath = writer_->add(iter);
		if (filepath.empty()) {
			return;
		}
		grid_.save_vtk(filepath);
		VtkUtils::add_cell_data(p_, "pressure", filepath, grid_.n_cells());
		VtkUtils::add_cell_data(k_, "k", filepath, grid_.n_cells());
		VtkUtils::add_cell_data(epsilon_, "epsilon", filepath, grid_.n_cells());
		VtkUtils::add_cell_data(nuT_, "nu_turb", filepath, grid_.n_cells());
		VtkUtils::add_cell_vector(u_, v_, "velocity", filepath, grid_.n_cells());
	}
}

std::vector<double> CavityFvmWorker::compute_dn_face(const std::vector<double>& x, const std::vector<double>& y) const {
	std::vector<double> ret(grid_.n_faces(), 0.0);
	for (size_t iface = 0; iface < grid_.n_faces(); ++iface) {
		auto [ci, cj] = collocations_.tab_face_colloc(iface);
		Vector normal = grid_.face_normal(iface);
		Vector u_face;
		if (collocations_.is_boundary_colloc(ci)) {
			u_face = Vector{x[cj], y[cj]};
		} else if (collocations_.is_boundary_colloc(cj)) {
			u_face = Vector{x[ci], y[ci]};
		} else {
			u_face = Vector{0.5 * (x[ci] + x[cj]), 0.5 * (y[ci] + y[cj])};
		}
		ret[iface] = u_face.x() * normal.x() * normal.x() + u_face.y() * normal.y() * normal.y();
	}
	return ret;
}

CsrMatrix CavityFvmWorker::assemble_convdiff_lhs(
		const std::vector<double>& u,
		const std::vector<double>& nu_face,
		const std::vector<double>& diag) const{
	std::vector<Vector> grad_u = grad_computer_->compute(u);

	// =============== Left side
	LodMatrix mat(vec_size());
	for (size_t iface = 0; iface < grid_.n_faces(); ++iface){
		size_t negative_colloc = collocations_.tab_face_colloc(iface)[0];
		size_t positive_colloc = collocations_.tab_face_colloc(iface)[1];
		double area = grid_.face_area(iface);
		double un = un_face_[iface];

		// - nu * Laplace(u)
		for (const std::pair<const size_t, double>& iter: dfdn_computer_.linear_combination(iface)){
			size_t column = iter.first;
			double coef = - nu_face[iface] * area * iter.second;
			mat.add_value(negative_colloc, column, coef);
			mat.add_value(positive_colloc, column, -coef);
		}

		// + nonlinear tvd convection
		{
			size_t i = negative_colloc; // upwind cell
			size_t j = positive_colloc; // downwind cell
			double coef = area * un;
			if (un < 0){
				std::swap(i, j);
				coef *= -1;
			};
			Vector cij = collocations_.points[j] - collocations_.points[i];

			double wi = tvd_upwind_weight(u, grad_u, cij, i, j);
			double wj = 1 - wi;
			mat.add_value(i, i,  wi*coef);
			mat.add_value(i, j,  wj*coef);
			mat.add_value(j, j, -wj*coef);
			mat.add_value(j, i, -wi*coef);
		}
	}
	// + diag
	if (diag.size() > 0){
		for (size_t icell = 0; icell < grid_.n_cells(); ++icell){
			mat.add_value(icell, icell, diag[icell] * grid_.cell_volume(icell));
		}
	}

	return mat.to_csr();
};

void CavityFvmWorker::prepare_velocity_slae() {
	// ============== left side
	std::vector<double> nu = compute_nueff_face();
	mat_u_ = assemble_convdiff_lhs(u_, nu);
	mat_v_ = assemble_convdiff_lhs(v_, nu);
	// ============== right side
	rhs_u_.resize(vec_size());
	rhs_v_.resize(vec_size());
	for (size_t icell = 0; icell < grid_.n_cells(); ++icell){
		rhs_u_[icell] = -grad_p_[icell].x() * grid_.cell_volume(icell);
		rhs_v_[icell] = -grad_p_[icell].y() * grid_.cell_volume(icell);
	}
	// ============== relax
	relax_slae(mat_u_, rhs_u_, u_, ALPHA_U, grid_.n_cells());
	relax_slae(mat_v_, rhs_v_, v_, ALPHA_U, grid_.n_cells());

	// ============== pressure coeffs
	for (size_t i=0; i<grid_.n_cells(); ++i){
		du_[i] = grid_.cell_volume(i) / mat_u_.value(i, i);
		dv_[i] = grid_.cell_volume(i) / mat_v_.value(i, i);
	}
        dn_face_ = compute_dn_face(du_, dv_);

	// bnd
	for (size_t icolloc: boundary_info_.all){
		mat_u_.set_unit_row(icolloc);
		mat_v_.set_unit_row(icolloc);
	}
	for (size_t icolloc: boundary_info_.nontop){
		rhs_u_[icolloc] = 0;
		rhs_v_[icolloc] = 0;
	}
	for (size_t icolloc: boundary_info_.top){
		rhs_u_[icolloc] = 1;
		rhs_v_[icolloc] = 0;
	}
}

std::vector<double> CavityFvmWorker::compute_p(const std::vector<double>& un_face) const {
	LodMatrix Ap(vec_size());
	std::vector<double> rhs(vec_size(), 0);
	// assemble slae
	for (size_t iface = 0; iface < grid_.n_faces(); ++iface) {
		auto [negative_colloc, positive_colloc] = collocations_.tab_face_colloc(iface);
		double area = grid_.face_area(iface);
		// matrix: div(d*grad(p))
		for (const std::pair<const size_t, double>& iter: dfdn_computer_.linear_combination(iface)) {
			size_t column = iter.first;
			double coef = -dn_face_[iface] * area * iter.second;
			Ap.add_value(negative_colloc, column, coef);
			Ap.add_value(positive_colloc, column, -coef);
		}
		// rhs: -div(u)
		{
			double coef = area * un_face[iface];
			if (!collocations_.is_boundary_colloc(negative_colloc)) {
				rhs[negative_colloc] -= coef;
			}
			if (!collocations_.is_boundary_colloc(positive_colloc)) {
				rhs[positive_colloc] += coef;
			}
		}
	}
	Ap.set_unit_row(0);
	rhs[0] = 0.0;
	// solve
	std::vector<double> p_tilde = p_;
	AmgcMatrixSolver::solve_slae(Ap.to_csr(), rhs, p_tilde);
	return p_tilde;
}

std::vector<double> CavityFvmWorker::compute_nueff_face(double sigma) const{
	// TODO: nueff_face = 1/Re  + nuT_face / sigma
	return std::vector<double>(grid_.n_faces(), 1.0/Re_);
}

std::vector<double> CavityFvmWorker::compute_G() const{
	// TODO
	return std::vector<double>(vec_size(), 0.0);
};

std::vector<double> CavityFvmWorker::compute_gamma() const{
	// TODO
	return std::vector<double>(vec_size(), 0.0);
}

std::vector<double> CavityFvmWorker::compute_nuT() const{
	// TODO
	return std::vector<double>(vec_size(), DELTA_NUT/Re_);
}

std::vector<double> CavityFvmWorker::compute_k(const std::vector<double>& G, const std::vector<double>& gamma) const{
	// left side
	std::vector<double> nu = compute_nueff_face(SIGMA_K);
	std::vector<double> diag(vec_size(), 0.0);
	for (size_t i=0; i<grid_.n_cells(); ++i) {
		diag[i] = gamma[i];
	}
	CsrMatrix mat = assemble_convdiff_lhs(k_, nu, diag);
	//  right side
	std::vector<double> rhs(vec_size(), 0.0);
	for (size_t icell = 0; icell < grid_.n_cells(); ++icell){
		rhs[icell] = G[icell] * grid_.cell_volume(icell);
	}
	relax_slae(mat, rhs, k_, ALPHA_K, grid_.n_cells());

	// solve
	std::vector<double> new_k(k_);
	AmgcMatrixSolver::solve_slae(mat, rhs, new_k);
	return new_k;
}

std::vector<double> CavityFvmWorker::compute_epsilon(const std::vector<double>& G, const std::vector<double>& gamma) const{
	std::vector<double> nu = compute_nueff_face(SIGMA_E);
	std::vector<double> diag(vec_size(), 0.0);
	for (size_t i=0; i<grid_.n_cells(); ++i) {
		diag[i] = C2 * gamma[i];
	}
	CsrMatrix mat = assemble_convdiff_lhs(epsilon_, nu, diag);
	// ============== right side
	std::vector<double> rhs(vec_size(), 0.0);
	for (size_t icell = 0; icell < grid_.n_cells(); ++icell){
		double v = C1 * G[icell] * gamma[icell];
		rhs[icell] = v * grid_.cell_volume(icell);
	}
	relax_slae(mat, rhs, epsilon_, ALPHA_E, grid_.n_cells());

	//boundary condition
	for (const auto& it: boundary_info_.bnd_cell_to_colloc){
		size_t icell = it.first;
		rhs[icell] = 0;
		mat.set_unit_row(icell);

		double w = 1.0 / it.second.size();
		
		for (size_t icolloc: it.second){
			//size_t iface = collocations_.face_index(icolloc);
			double eps_i = 0; // TODO
			// on boundary
			mat.set_unit_row(icolloc);
			rhs[icolloc] = eps_i;
			// on near-boundary cell
			rhs[icell] = w * eps_i;
		}
	}
	
	// solve
	std::vector<double> new_e(epsilon_);
	AmgcMatrixSolver::solve_slae(mat, rhs, new_e);
	return new_e;
}

} // namespace

TEST_CASE("Cavity 2D, FVM-SIMPLE algorithm, k-epsilon", "[cavity-fvm-ke]") {
	std::cout << std::endl << "--- cfd25_test [cavity-fvm-ke] --- " << std::endl;

	// problem parameters
	double Re = 10'000;
	size_t max_it = 10000;
	double eps = 1e-1;

	// worker initialization
	//auto grid = RegularGrid2D(0, 1, 0, 1, 30, 30);
	auto grid = UnstructuredGrid2D::vtk_read(test_directory_file("tetragrid_500.vtk"));
	CavityFvmWorker worker(grid, Re);
	worker.initialize_saver("cavity-fvm-ke", 5);
	worker.save_current_fields(0);

	size_t it = 0;
	double nrm = 0;
	for (it = 1; it < max_it; ++it) {
		nrm = worker.step();
		worker.save_current_fields(it);
		std::cout << it << ": " << nrm << std::endl;
		// break inner iterations if residual is low enough
		if (nrm < eps) {
			break;
		} else if (it == max_it - 1) {
			std::cout << "WARNING: internal SIMPLE interations not converged with nrm = " << nrm << std::endl;
		}
	}
	CHECK(it == 17);
	CHECK(nrm == Approx(0.0922024602));
}
