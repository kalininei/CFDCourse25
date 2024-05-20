#include "triangle_linear.hpp"

using namespace cfd;

///////////////////////////////////////////////////////////////////////////////
// Geometry
///////////////////////////////////////////////////////////////////////////////
TriangleLinearGeometry::TriangleLinearGeometry(Point p0, Point p1, Point p2): _p0(p0), _p1(p1), _p2(p2){
	_jac.modj = vector_abs(cross_product(p1 - p0, p2 - p0));
	_jac.j11 = p1.x() - p0.x();
	_jac.j12 = p2.x() - p0.x();
	_jac.j21 = p1.y() - p0.y();
	_jac.j22 = p2.y() - p0.y();
	_jac.j31 = p1.z() - p0.z();
	_jac.j32 = p2.z() - p0.z();
}

JacobiMatrix TriangleLinearGeometry::jacobi(Point xi) const{
	return _jac;
}

Point TriangleLinearGeometry::to_physical(Point xi_) const{
	double xi = xi_.x();
	double eta = xi_.y();
	return (1 - xi - eta) * _p0 + xi * _p1 + eta * _p2;
}

Point TriangleLinearGeometry::parametric_center() const{
	return {1.0/3.0, 1.0/3.0};
}

///////////////////////////////////////////////////////////////////////////////
// Basis
///////////////////////////////////////////////////////////////////////////////
size_t TriangleLinearBasis::size() const {
	return 3;
}

std::vector<Point> TriangleLinearBasis::parametric_reference_points() const {
	return {Point(0, 0), Point(1, 0), Point(0, 1)};
}

std::vector<BasisType> TriangleLinearBasis::basis_types() const {
	return {BasisType::Nodal, BasisType::Nodal, BasisType::Nodal};
}

std::vector<double> TriangleLinearBasis::value(Point xi_) const {
	double xi = xi_.x();
	double eta = xi_.y();
	return { 1 - xi - eta, xi, eta };
}

std::vector<Vector> TriangleLinearBasis::grad(Point xi) const {
	return { Vector(-1, -1), Vector(1, 0), Vector(0, 1) };
}

///////////////////////////////////////////////////////////////////////////////
// Integrals
///////////////////////////////////////////////////////////////////////////////
TriangleLinearIntegrals::TriangleLinearIntegrals(JacobiMatrix jac): _jac(jac) { };

std::vector<double> TriangleLinearIntegrals::mass_matrix() const {
	double s0 = _jac.modj/12.0;
	double s1 = _jac.modj/24.0;
	return {s0, s1, s1,
	        s1, s0, s1,
	        s1, s1, s0};
}

std::vector<double> TriangleLinearIntegrals::load_vector() const {
	double s0 = _jac.modj/6;
	return {s0, s0, s0};
}

std::vector<double> TriangleLinearIntegrals::stiff_matrix() const {
	double c= 0.5 / _jac.modj;
	double j11 = _jac.j11;
	double j12 = _jac.j12;
	double j21 = _jac.j21;
	double j22 = _jac.j22;

	double s00 = c*(j22*j22-2*j21*j22+j21*j21+j12*j12-2*j11*j12+j11*j11);
	double s01 = -c*(j22*j22-j21*j22+j12*j12-j11*j12);
	double s02 = c*(j21*j22-j21*j21+j11*j12-j11*j11);
	double s11 = c*(j22*j22+j12*j12);
	double s12 = -c*(j21*j22+j11*j12);
	double s22 = c*(j21*j21+j11*j11);

	return {s00, s01, s02,
		s01, s11, s12,
		s02, s12, s22};
}
