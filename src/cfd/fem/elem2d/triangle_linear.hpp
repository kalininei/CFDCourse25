#ifndef __CFD_FEM_TRIANGLE_LINEAR_HPP__
#define __CFD_FEM_TRIANGLE_LINEAR_HPP__

#include "cfd/fem/fem_element.hpp"

namespace cfd{

///////////////////////////////////////////////////////////////////////////////
// Geometry
///////////////////////////////////////////////////////////////////////////////
class TriangleLinearGeometry: public IElementGeometry{
public:
	TriangleLinearGeometry(Point p0, Point p1, Point p2);

	JacobiMatrix jacobi(Point xi) const override;
	Point to_physical(Point xi) const override;
	Point to_parametric(Point p) const override;
	Point parametric_center() const override;
private:
	Point _p0, _p1, _p2;
	JacobiMatrix _jac;
};

///////////////////////////////////////////////////////////////////////////////
// Basis
///////////////////////////////////////////////////////////////////////////////
class TriangleLinearBasis: public IElementBasis{
public:
	size_t size() const override;
	std::vector<Point> parametric_reference_points() const override;
	std::vector<BasisType> basis_types() const override;
	std::vector<double> value(Point xi) const override;
	std::vector<Vector> grad(Point xi) const override;
};

///////////////////////////////////////////////////////////////////////////////
// Integrals
///////////////////////////////////////////////////////////////////////////////
class TriangleLinearIntegrals: public IElementIntegrals{
public:
	TriangleLinearIntegrals(JacobiMatrix jac);
	std::vector<double> mass_matrix() const override;
	std::vector<double> load_vector() const override;
	std::vector<double> stiff_matrix() const override;
private:
	const JacobiMatrix _jac;
};

}

#endif
