#ifndef __CFD_FEM_TRIANGLE_CUBIC_HPP__
#define __CFD_FEM_TRIANGLE_CUBIC_HPP__

#include "cfd/fem/fem_element.hpp"

namespace cfd{

///////////////////////////////////////////////////////////////////////////////
// Basis, 10 node cubic
///////////////////////////////////////////////////////////////////////////////
class TriangleCubicBasis: public IElementBasis{
public:
	size_t size() const override;
	std::vector<Point> parametric_reference_points() const override;
	std::vector<BasisType> basis_types() const override;
	std::vector<double> value(Point xi) const override;
	std::vector<Vector> grad(Point xi) const override;
};

///////////////////////////////////////////////////////////////////////////////
// Basis, 9 node cubic
///////////////////////////////////////////////////////////////////////////////
class TriangleCubic9Basis: public IElementBasis{
public:
	size_t size() const override;
	std::vector<Point> parametric_reference_points() const override;
	std::vector<BasisType> basis_types() const override;
	std::vector<double> value(Point xi) const override;
	std::vector<Vector> grad(Point xi) const override;
};

///////////////////////////////////////////////////////////////////////////////
// Basis, 9 node cubic no11
///////////////////////////////////////////////////////////////////////////////
class TriangleCubicNo11Basis: public IElementBasis{
public:
	size_t size() const override;
	std::vector<Point> parametric_reference_points() const override;
	std::vector<BasisType> basis_types() const override;
	std::vector<double> value(Point xi) const override;
	std::vector<Vector> grad(Point xi) const override;
};

///////////////////////////////////////////////////////////////////////////////
// Basis, hermite cubic
///////////////////////////////////////////////////////////////////////////////
class TriangleHermiteBasis: public IElementBasis{
public:
	TriangleHermiteBasis(std::shared_ptr<const IElementGeometry> geom);

	size_t size() const override;
	std::vector<Point> parametric_reference_points() const override;
	std::vector<BasisType> basis_types() const override;
	std::vector<double> value(Point xi) const override;
	std::vector<Vector> grad(Point xi) const override;
private:
	std::shared_ptr<const IElementGeometry> _geom;
};

}

#endif
