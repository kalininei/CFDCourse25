#include "fvm_dpdn_boundary.hpp"
#include <numeric>
#include <list>
#include "cfd/debug/printer.hpp"

using namespace cfd;

FvmDpDnComputer2D::FvmDpDnComputer2D(double Re, double alpha_p, double tau, double d,
                                     const IGrid& grid, const FvmExtendedCollocations& colloc):
		_dfdn(grid, colloc),
		_Re(Re), _alpha_p(alpha_p), _tau(tau), _d(d){

	_n = colloc.face_collocations.size();
	_values.resize(_n, 0.0);

	// assemble main geometry
	_face_indices.push_back(colloc.face_index(colloc.face_collocations[0]));
	_original_indices.resize(_n, 0);
	_revert_coef.push_back(1.0);
	size_t pfirst = grid.tab_face_point(_face_indices[0])[0];
	size_t pcur = grid.tab_face_point(_face_indices[0])[1];
	if (grid.tab_face_cell(_face_indices[0])[0] == INVALID_INDEX){
		_revert_coef[0] *= -1;
		std::swap(pfirst, pcur);
	}
	std::list<size_t> indices;
	for (size_t i=1; i<_n; ++i) indices.push_back(i);
	while (indices.size() > 0){
		bool found = false;
		for (auto it=indices.begin(); it != indices.end(); ++it){
			size_t iface = colloc.face_index(colloc.face_collocations[*it]);
			size_t p0 = grid.tab_face_point(iface)[0];
			size_t p1 = grid.tab_face_point(iface)[1];
			if (p0 == pcur || p1 == pcur){
				_face_indices.push_back(iface);
				_original_indices[_face_indices.size() - 1] = *it;
				if (p0 == pcur){
					_revert_coef.push_back(1.0);
					pcur = p1;
				} else {
					_revert_coef.push_back(-1.0);
					pcur = p0;
				}
				indices.erase(it);
				found = true;
				break;
			}
		}

		if (!found){
			_THROW_INTERNAL_ERROR_;
		}
	}
	if (pcur != pfirst){
		_THROW_INTERNAL_ERROR_;
	}

	// assemble helper geometry
	for (size_t i=0; i<_n; ++i){
		size_t iface = _face_indices[i];
		Point p0 = grid.point(grid.tab_face_point(iface)[0]);
		Point p1 = grid.point(grid.tab_face_point(iface)[1]);
		if (_revert_coef[i] < 0){
			std::swap(p0, p1);
		}
		_face_tangent.push_back((p1-p0)/vector_abs(p1-p0));
		_inext.push_back( (i==_n-1) ? 0 : i + 1);
		_iprev.push_back( (i==0) ? _n-1 : i - 1);
		_face_areas.push_back(grid.face_area(_face_indices[i]));
		_face_hnext.push_back(0.5*(grid.face_area(_face_indices[i]) + grid.face_area(_face_indices[_inext[i]])));
		_face_hprev.push_back(0.5*(grid.face_area(_face_indices[i]) + grid.face_area(_face_indices[_iprev[i]])));
	}
	_total_area = std::accumulate(_face_areas.begin(), _face_areas.end(), 0.0);
	
	// assemble lhs
	LodMatrix mat(_values.size());
	for (size_t i=0; i<_n; ++i){
		// diagonal
		mat.add_value(i, i, _alpha_p * _face_areas[i]);
		// diffusion
		double coef = _tau * _d / _Re;
		double coef_next = coef/ _face_hnext[i];
		double coef_prev = coef/ _face_hprev[i];
		mat.add_value(i, _iprev[i], -coef_prev);
		mat.add_value(i, _inext[i], -coef_next);
		mat.add_value(i, i, coef_prev + coef_next);
	}

	//dbg::print(mat.to_csr());
	_solver.set_matrix(mat.to_csr());
}

void FvmDpDnComputer2D::solve(const std::vector<double>& pold, const std::vector<double>& ustar, const std::vector<double>& vstar){
	//assemble rhs
	std::vector<double> rhs(_n, 0);
	for (size_t i=0; i<_n; ++i){
		size_t iface = _face_indices[i];
		size_t iface_next = _face_indices[_inext[i]];
		size_t iface_prev = _face_indices[_iprev[i]];
	
		// -dpold/dn
		rhs[i] -= _revert_coef[i]*_face_areas[i]*_dfdn.compute(iface, pold);

		// -1/Re*0.5*((du_s/dn)_next - (du_s/dn)_prev)
		double dus_dn_next = 0;
		Vector tangent_next = _face_tangent[_inext[i]];
		for (auto it: _dfdn.linear_combination(iface_next)){
			size_t icolloc = it.first;
			double value = it.second;
			double us = dot_product(Vector{ustar[icolloc], vstar[icolloc]}, tangent_next);
			dus_dn_next += us * value * _revert_coef[_inext[i]];
		}
		rhs[i] -= 0.5/_Re*dus_dn_next;

		double dus_dn_prev = 0;
		Vector tangent_prev = _face_tangent[_iprev[i]];
		for (auto it: _dfdn.linear_combination(iface_prev)){
			size_t icolloc = it.first;
			double value = it.second;
			double us = dot_product(Vector{ustar[icolloc], vstar[icolloc]}, tangent_prev);
			dus_dn_prev += us * value * _revert_coef[_iprev[i]];
		}
		rhs[i] += 0.5/_Re*dus_dn_prev;
	}

	std::vector<double> dpdn;
	_solver.solve(rhs, dpdn);

	// solution condition
	double sum = 0;
	for (size_t i=0; i<_n; ++i){
		sum += dpdn[i] * _face_areas[i];
	}
	double delta = sum/_total_area;

	// assign
	for (size_t i=0; i<_n; ++i){
		_values[_original_indices[i]] = (dpdn[i] - delta);
	}
}

double FvmDpDnComputer2D::get_dpdn(size_t ic) const{
	return _values[ic];
}
