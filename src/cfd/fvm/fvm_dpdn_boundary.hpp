#ifndef __FVM_DPDN_BOUNDARY_HPP__
#define __FVM_DPDN_BOUNDARY_HPP__

#include "cfd/fvm/fvm_assembler.hpp"
#include "cfd/mat/sparse_matrix_solver.hpp"

namespace cfd{

class FvmDpDnComputer2D{
public:
	FvmDpDnComputer2D(double Re, double alpha_p, double tau, double d,
	                  const IGrid& grid, const FvmExtendedCollocations& collocublic);

	void solve(const std::vector<double>& p, const std::vector<double>& ustar, const std::vector<double>& vstar);
	double get_dpdn(size_t ic) const;
private:
	const FvmFacesDn _dfdn;
	const double _Re;
	const double _alpha_p;
	const double _tau;
	const double _d;
	size_t _n;
	std::vector<double> _values;
	AmgcMatrixSolver _solver;
	std::vector<double> _revert_coef;
	std::vector<double> _face_areas;
	std::vector<size_t> _face_indices;
	std::vector<size_t> _inext;
	std::vector<size_t> _iprev;
	std::vector<double> _face_hnext;
	std::vector<double> _face_hprev;
	std::vector<size_t> _original_indices;
	std::vector<Vector> _face_tangent;
	double _total_area;
};

}

#endif
