#include "fem_element.hpp"
#include <limits>

using namespace cfd;


struct IElementIntegrals::OperandArg::Cache{
	std::vector<double> phi;
	std::vector<Vector> grad_phi;

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
