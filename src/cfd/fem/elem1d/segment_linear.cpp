#include "segment_linear.hpp"

using namespace cfd;

///////////////////////////////////////////////////////////////////////////////
// Geometry
///////////////////////////////////////////////////////////////////////////////
SegmentLinearGeometry::SegmentLinearGeometry(Point p0, Point p1): _p0(p0), _p1(p1){
	double modj = vector_abs(p1 - p0)/2;
	_jac.j11 = modj;
	_jac.modj = modj;
}

JacobiMatrix SegmentLinearGeometry::jacobi(Point xi) const{
	return _jac;
}

Point SegmentLinearGeometry::to_physical(Point xi) const{
	double t = (xi.x() + 1)/2.0;
	return (1 - t) * _p0 + t * _p1;
}

Point SegmentLinearGeometry::parametric_center() const{
	return {0};
}

///////////////////////////////////////////////////////////////////////////////
// Basis
///////////////////////////////////////////////////////////////////////////////
size_t SegmentLinearBasis::size() const {
	return 2;
}

std::vector<Point> SegmentLinearBasis::parametric_reference_points() const {
	return {Point(-1), Point(1)};
}

std::vector<BasisType> SegmentLinearBasis::basis_types() const {
	return {BasisType::Nodal, BasisType::Nodal};
}

std::vector<double> SegmentLinearBasis::value(Point xi_) const {
	double xi = xi_.x();
	return { (1 - xi)/2, (1 + xi)/2 };
}

std::vector<Vector> SegmentLinearBasis::grad(Point xi) const {
	return { Vector{-0.5}, Vector{0.5} };
}

///////////////////////////////////////////////////////////////////////////////
// Integrals
///////////////////////////////////////////////////////////////////////////////
SegmentLinearIntegrals::SegmentLinearIntegrals(JacobiMatrix jac): _len(2*jac.modj) { };

std::vector<double> SegmentLinearIntegrals::mass_matrix() const {
	double v = _len/6; 
	return {2*v,   v,
	          v, 2*v};
}

std::vector<double> SegmentLinearIntegrals::load_vector() const {
	double s0 = _len/2;
	return {s0, s0};
}

std::vector<double> SegmentLinearIntegrals::stiff_matrix() const {
	double s = 1/_len;
	return {s, -s,
		-s, s};
}
