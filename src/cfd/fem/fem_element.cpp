#include "fem_element.hpp"
#include <limits>

using namespace cfd;


struct IElementIntegrals::OperandArg::Cache{
	std::vector<double> phi;
	std::vector<Vector> grad_phi;
	std::vector<double> laplace;

	double has_jac = false;
	JacobiMatrix jac;
};

IElementIntegrals::OperandArg::OperandArg(const IElementGeometry* geom, const IElementBasis* basis, Point xi)
	: geom(geom),
	  basis(basis),
	  xi(xi),
	  _pcache(new Cache()){ }

IElementIntegrals::OperandArg::~OperandArg(){};

double IElementIntegrals::OperandArg::phi(size_t i) const{
	if (_pcache->phi.size() == 0){
		_pcache->phi = basis->value(xi);
	}
	return _pcache->phi[i];
}

Vector IElementIntegrals::OperandArg::grad_phi(size_t i) const{
	if (_pcache->grad_phi.size() == 0){
		std::vector<Vector> grad_xi = basis->grad(xi);
		for (size_t i=0; i<grad_xi.size(); ++i){
			_pcache->grad_phi.push_back(gradient_to_physical(*jacobi(), grad_xi[i]));
		}
	}
	return _pcache->grad_phi[i];
}

double IElementIntegrals::OperandArg::laplace(size_t i) const{
	if (_pcache->laplace.size() == 0){
		_pcache->laplace.resize(basis->size(), 0);
		double h = 1e-6;  // <- TODO: Compute from jacobian

		Vector x = geom->to_physical(xi);
		Vector xi1_minus = geom->to_parametric(x - Vector{h, 0, 0});
		Vector xi1_plus =  geom->to_parametric(x + Vector{h, 0, 0});
		Vector xi2_minus = geom->to_parametric(x - Vector{0, h, 0});
		Vector xi2_plus =  geom->to_parametric(x + Vector{0, h, 0});
		Vector xi3_minus = geom->to_parametric(x - Vector{0, 0, h});
		Vector xi3_plus =  geom->to_parametric(x + Vector{0, 0, h});

		std::vector<Vector> grad1_minus = basis->grad(xi1_minus);
		std::vector<Vector> grad1_plus  = basis->grad(xi1_plus);
		std::vector<Vector> grad2_minus = basis->grad(xi2_minus);
		std::vector<Vector> grad2_plus  = basis->grad(xi2_plus);
		std::vector<Vector> grad3_minus = basis->grad(xi3_minus);
		std::vector<Vector> grad3_plus  = basis->grad(xi3_plus);

		for (size_t i=0; i<basis->size(); ++i){
			// use finite difference to compute divergence.
			// TODO: should be done using metric tensor and Hesse matrix
			double ddx0 = gradient_to_physical(*jacobi(), grad1_minus[i]).x();
			double ddx1 = gradient_to_physical(*jacobi(), grad1_plus[i]).x();
			double ddy0 = gradient_to_physical(*jacobi(), grad2_minus[i]).y();
			double ddy1 = gradient_to_physical(*jacobi(), grad2_plus[i]).y();
			double ddz0 = gradient_to_physical(*jacobi(), grad3_minus[i]).z();
			double ddz1 = gradient_to_physical(*jacobi(), grad3_plus[i]).z();
			_pcache->laplace[i] = (ddx1 - ddx0)/(2*h) + (ddy1 - ddy0)/(2*h) + (ddz1 - ddz0)/(2*h);
		}
	}
	return _pcache->laplace[i];
}

const JacobiMatrix* IElementIntegrals::OperandArg::jacobi() const{
	if (!_pcache->has_jac){
		_pcache->jac = geom->jacobi(xi);
		_pcache->has_jac = true;
	}
	return &(_pcache->jac);
}

double IElementIntegrals::OperandArg::modj() const{
	return jacobi()->modj;
}

double IElementIntegrals::OperandArg::interpolate(const std::vector<double>& f) const{
	_THROW_NOT_IMP_;
}

Vector IElementIntegrals::OperandArg::interpolate(const std::vector<Vector>& f) const{
	Vector ret;
	for (size_t i=0; i<basis->size(); ++i){
		ret += phi(i) * f[i];
	}
	return ret;
}

double IElementIntegrals::OperandArg::divergence(const std::vector<Vector>& f) const{
	double ret = 0.0;
	for (size_t i=0; i<basis->size(); ++i){
		Vector dphi = grad_phi(i);
		ret += (f[i].x() * dphi.x() + f[i].y() * dphi.y() + f[i].z() * dphi.z());
	}
	return ret;
}
