#include "cfd25_test.hpp"
#include "cfd/grid/grid1d.hpp"
#include "cfd/mat/csrmat.hpp"
#include "cfd/mat/sparse_matrix_solver.hpp"
#include "cfd/grid/vtk.hpp"
#include "cfd/mat/lodmat.hpp"

using namespace cfd;

class ATestTransport1Worker{
public:
	static double init_solution(double x){
		//return (x >= -0.1 && x <= 0.1) ? 1.0 : 0.0;
		constexpr double sigma = 0.1;
		return exp(-x*x/sigma/sigma);
	}
	double exact_solution(double x){
		return init_solution(x-_time);
	}

	ATestTransport1Worker(size_t n_cells): _grid(0, 1, n_cells), _h(1.0/n_cells), _u(_grid.n_points()){
		for (size_t i=0; i<_grid.n_points(); ++i){
			_u[i] = init_solution(_grid.point(i).x());
		}
	}

	double step(double tau){
		_time += tau;
		impl_step(tau);
		return compute_norm2();
	}

	void save_vtk(const std::string& filename){
		// save grid
		_grid.save_vtk(filename);
		
		// save numerical solution
		VtkUtils::add_point_data(_u, "numerical", filename);

		// save exact solution
		std::vector<double> exact(_grid.n_points());
		for (size_t i=0; i<_grid.n_points(); ++i){
			exact[i] = exact_solution(_grid.point(i).x());
		}
		VtkUtils::add_point_data(exact, "exact", filename);
	}

	double current_time() const {
		return _time;
	}

	double compute_norm2(){
		// weights
		std::vector<double> w(_grid.n_points(), _h);
		w[0] = w[_grid.n_points()-1] = _h/2;

		// sum
		double sum = 0;
		for (size_t i=0; i<_grid.n_points(); ++i){
			double diff = _u[i] - exact_solution(_grid.point(i).x());
			sum += w[i]*diff*diff;
		}

		double len = _grid.point(_grid.n_points()-1).x() - _grid.point(0).x();
		return std::sqrt(sum / len);
	}
protected:
	Grid1D _grid;
	double _h;
	std::vector<double> _u;
	double _time = 0;

private:
	virtual void impl_step(double tau) = 0;
};

///////////////////////////////////////////////////////////////////////////////
// Explicit transport 1D solver
///////////////////////////////////////////////////////////////////////////////

class TestTransport1WorkerExplicit: public ATestTransport1Worker{
public:
	TestTransport1WorkerExplicit(size_t n_cells): ATestTransport1Worker(n_cells){}

private:
	void impl_step(double tau) override {
		std::vector<double> u_old(_u);
		_u[0] = exact_solution(_grid.point(0).x());
		for (size_t i=1; i<_grid.n_points(); ++i){
			_u[i] = u_old[i] - tau/_h*(u_old[i] - u_old[i-1]);
		}
	}
};

TEST_CASE("Transport 1D solver, explicit", "[transport1-explicit]"){
	std::cout << std::endl << "--- cfd_test [transport1-explicit] --- " << std::endl;
	// parameters
	const double tend = 0.5;
	const double V = 1.0;
	const double L = 1.0;
	size_t n_cells = 100;
	double Cu = 0.9;
	double h = L/n_cells;
	double tau = Cu * h / V;

	// solver
	TestTransport1WorkerExplicit worker(n_cells);

	// saver
	VtkUtils::TimeSeriesWriter writer("transport1-explicit");
	std::string out_filename = writer.add(0);
	worker.save_vtk(out_filename);

	double norm = 0;
	while (worker.current_time() < tend - 1e-6) {
		// solve problem
		norm = worker.step(tau);
		// export solution to vtk
		out_filename = writer.add(worker.current_time());
		worker.save_vtk(out_filename);
	};
	std::cout << 1.0/tau << " " << norm << std::endl;
	CHECK(norm == Approx(0.0138123932).margin(1e-5));
}

///////////////////////////////////////////////////////////////////////////////
// Implicit transport 1D solver
///////////////////////////////////////////////////////////////////////////////

class TestTransport1WorkerImplicit: public ATestTransport1Worker{
public:
	TestTransport1WorkerImplicit(size_t n_cells): ATestTransport1Worker(n_cells){}

private:
	void impl_step(double tau) override {
		AmgcMatrixSolver& slv = build_solver(tau);
		std::vector<double> rhs = build_rhs(tau);
		slv.solve(rhs, _u);
	}

	AmgcMatrixSolver _solver;
	double _last_used_tau = 0;

	AmgcMatrixSolver& build_solver(double tau){
		if (_last_used_tau != tau){
			CsrMatrix mat = build_lhs(tau);
			_solver.set_matrix(mat);
			_last_used_tau = tau;
		}
		return _solver;
	}

	virtual CsrMatrix build_lhs(double tau){
		LodMatrix mat(_u.size());
		mat.set_value(0, 0, 1.0);
		mat.set_value(_u.size()-1, _u.size()-1, 1.0);
		double diag = 1.0 + tau/_h;
		double nondiag = -tau/_h;
		for (size_t i=1; i<_u.size()-1; ++i){
			mat.set_value(i, i, diag);
			mat.set_value(i, i-1, nondiag);
		}
		return mat.to_csr();
	}

	virtual std::vector<double> build_rhs(double tau){
		std::vector<double> rhs(_u);
		rhs[0] = exact_solution(_grid.point(0).x());
		rhs.back() = exact_solution(_grid.point(_grid.n_points()-1).x());
		return rhs;
	}
};

TEST_CASE("Transport 1D solver, implicit", "[transport1-implicit]"){
	std::cout << std::endl << "--- cfd_test [transport1-implicit] --- " << std::endl;
	const double tend = 0.5;
	const double V = 1.0;
	const double L = 1.0;
	size_t n_cells = 100;
	double Cu = 0.9;

	double h = L/n_cells;
	double tau = Cu * h / V;
	TestTransport1WorkerImplicit worker(n_cells);
	VtkUtils::TimeSeriesWriter writer("transport1-implicit");
	double norm;
	worker.save_vtk(writer.add(0));
	while (worker.current_time() < tend - 1e-6) {
		norm = worker.step(tau);
		worker.save_vtk(writer.add(worker.current_time()));
	};
	std::cout << 1.0/tau << " " << norm << std::endl;
	CHECK(norm == Approx(0.137664).margin(1e-5));
}

///////////////////////////////////////////////////////////////////////////////
// Crank-Nicolson transport 1D solver
///////////////////////////////////////////////////////////////////////////////

class TestTransport1WorkerCN: public TestTransport1WorkerImplicit{
public:
	TestTransport1WorkerCN(size_t n_cells): TestTransport1WorkerImplicit(n_cells){}
private:
	CsrMatrix build_lhs(double tau) override{
		LodMatrix mat(_u.size());
		mat.set_value(0, 0, 1.0);
		mat.set_value(_u.size()-1, _u.size()-1, 1.0);
		double diag = 1.0 + 0.5*tau/_h;
		double nondiag = -0.5*tau/_h;
		for (size_t i=1; i<_u.size()-1; ++i){
			mat.set_value(i, i, diag);
			mat.set_value(i, i-1, nondiag);
		}
		return mat.to_csr();
	}

	std::vector<double> build_rhs(double tau) override{
		std::vector<double> rhs(_u);
		rhs[0] = exact_solution(_grid.point(0).x());
		rhs.back() = exact_solution(_grid.point(_grid.n_points()-1).x());
		for (size_t i=1; i<rhs.size()-1; ++i){
			rhs[i] -= 0.5 * tau / _h * (_u[i] - _u[i-1]);
		}
		return rhs;
	}
};

TEST_CASE("Transport 1D solver, Crank-Nicolson", "[transport1-cn]"){
	std::cout << std::endl << "--- cfd_test [transport1-cn] --- " << std::endl;
	const double tend = 0.5;
	const double V = 1.0;
	const double L = 1.0;
	size_t n_cells = 100;
	double Cu = 0.9;

	double h = L/n_cells;
	double tau = Cu * h / V;
	TestTransport1WorkerCN worker(n_cells);
	VtkUtils::TimeSeriesWriter writer("transport1-cn");
	double norm;
	worker.save_vtk(writer.add(0));
	while (worker.current_time() < tend - 1e-6) {
		norm = worker.step(tau);
		worker.save_vtk(writer.add(worker.current_time()));
	};
	std::cout << 1.0/tau << " " << norm << std::endl;
	CHECK(norm == Approx(0.0937748).margin(1e-5));
}

TEST_CASE("b", "[b]"){
	size_t n = 50;
	size_t maxit = 1000;
	auto exact = [](double x){ return 1*sin(2*M_PI*x) + 0.2 * sin(20*M_PI*x); };
	auto rhs = [](double x){ return 4*M_PI*M_PI*sin(2*M_PI*x) + 0.2 * 400*M_PI*M_PI*sin(20*M_PI*x); };

	std::vector<double> x(n + 1, 0);
	x[0] = exact(0);
	x[x.size()-1] = exact(1);
	double h = 1.0 / n;
	double c = 1.0 / h / h;
	std::vector<double> x_new = x;

	Grid1D grid(0, 1, n);
	VtkUtils::TimeSeriesWriter writer("convergence1");
	writer.set_time_step(10);

	for (size_t it=0; it<maxit; ++it){
		std::string fn = writer.add(it);
		if (!fn.empty()){
			grid.save_vtk(fn);
			VtkUtils::add_point_data(x, "data", fn);
		}

		std::cout << it << std::endl;
		// Jacobi
		// Seidel
		for (size_t i=1; i<x.size()-1; ++i){
			double coo = grid.point(i).x();
			x[i] = (rhs(coo) + c * (x[i-1] + x[i+1])) / (2*c);
		}
		//std::swap(x_new, x);
	}
}


