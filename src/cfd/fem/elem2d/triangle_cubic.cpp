#include "triangle_cubic.hpp"

using namespace cfd;

///////////////////////////////////////////////////////////////////////////////
// Basis, 10 node cubic
///////////////////////////////////////////////////////////////////////////////
size_t TriangleCubicBasis::size() const{
	return 10;
}

std::vector<Point> TriangleCubicBasis::parametric_reference_points() const{
	return {
		{0, 0},
		{1, 0},
		{0, 1},
		{1.0/3.0, 0},
		{2.0/3.0, 0},
		{2.0/3.0, 1.0/3.0},
		{1.0/3.0, 2.0/3.0},
		{0, 2.0/3.0},
		{0, 1.0/3.0},
		{1.0/3.0, 1.0/3.0}
	};
}

std::vector<BasisType> TriangleCubicBasis::basis_types() const{
	return std::vector<BasisType>(size(), BasisType::Nodal);
}

std::vector<double> TriangleCubicBasis::value(Point xi) const{
	double x = xi.x();
	double y = xi.y();
	return {
		0.5*(-9*y*y*y+(18-27*x)*y*y+(-27*x*x+36*x-11)*y-9*x*x*x+18*x*x-11*x+2),
		0.5*(9*x*x*x-9*x*x+2*x),
		0.5*(9*y*y*y-9*y*y+2*y),
		0.5*(27*x*y*y+(54*x*x-45*x)*y+27*x*x*x-45*x*x+18*x),
		0.5*((9*x-27*x*x)*y-27*x*x*x+36*x*x-9*x),
		0.5*((27*x*x-9*x)*y),
		0.5*(27*x*y*y-9*x*y),
		0.5*(-27*y*y*y+(36-27*x)*y*y+(9*x-9)*y),
		0.5*(27*y*y*y+(54*x-45)*y*y+(27*x*x-45*x+18)*y),
		0.5*((54*x-54*x*x)*y-54*x*y*y),
	};

}

std::vector<Vector> TriangleCubicBasis::grad(Point xi) const{
	double x = xi.x();
	double y = xi.y();
	return {
		0.5*Vector{-27*y*y+(36-54*x)*y-27*x*x+36*x-11, -27*y*y+(36-54*x)*y-27*x*x+36*x-11},
		0.5*Vector{27*x*x-18*x+2,                      0},
		0.5*Vector{0,                                  27*y*y-18*y+2},
		0.5*Vector{27*y*y+(108*x-45)*y+81*x*x-90*x+18, 54*x*y+54*x*x-45*x},
		0.5*Vector{(9-54*x)*y-81*x*x+72*x-9,           9*x-27*x*x},
		0.5*Vector{(54*x-9)*y,                         27*x*x-9*x},
		0.5*Vector{27*y*y-9*y,                         54*x*y-9*x},
		0.5*Vector{9*y-27*y*y,                         -81*y*y+(72-54*x)*y+9*x-9},
		0.5*Vector{54*y*y+(54*x-45)*y,                 81*y*y+(108*x-90)*y+27*x*x-45*x+18},
		0.5*Vector{(54-108*x)*y-54*y*y,                -108*x*y-54*x*x+54*x}
	};
}

///////////////////////////////////////////////////////////////////////////////
// Basis, 9 node cubic
///////////////////////////////////////////////////////////////////////////////
size_t TriangleCubic9Basis::size() const{
	return 9;
}

std::vector<Point> TriangleCubic9Basis::parametric_reference_points() const{
	return {
		{0, 0},
		{1, 0},
		{0, 1},
		{1.0/3.0, 0},
		{2.0/3.0, 0},
		{2.0/3.0, 1.0/3.0},
		{1.0/3.0, 2.0/3.0},
		{0, 2.0/3.0},
		{0, 1.0/3.0}
	};
}

std::vector<BasisType> TriangleCubic9Basis::basis_types() const{
	return std::vector<BasisType>(size(), BasisType::Nodal);
}

std::vector<double> TriangleCubic9Basis::value(Point xi) const{
	double x = xi.x();
	double y = xi.y();
	return {
		0.25*(-(18*y*y*y)+(36-18*x)*y*y+(-(18*x*x)+36*x-22)*y-18*x*x*x+36*x*x-22*x+4),
		0.25*(9*x*y*y+(9*x*x-9*x)*y+18*x*x*x-18*x*x+4*x),
		0.25*(18*y*y*y+(9*x-18)*y*y+(9*x*x-9*x+4)*y),
		0.25*((54*x*x-36*x)*y+54*x*x*x-90*x*x+36*x),
		0.25*((18*x-54*x*x)*y-54*x*x*x+72*x*x-18*x),
		0.25*((27*x*x+9*x)*y-27*x*y*y),
		0.25*(27*x*y*y+(9*x-27*x*x)*y),
		0.25*(-(54*y*y*y)+(72-54*x)*y*y+(18*x-18)*y),
		0.25*(54*y*y*y+(54*x-90)*y*y+(36-36*x)*y)
	};

}

std::vector<Vector> TriangleCubic9Basis::grad(Point xi) const{
	double x = xi.x();
	double y = xi.y();
	return {
		0.25*Vector{-(18*y*y)+(36-36*x)*y-54*x*x+72*x-22, -(54*y*y)+(72-36*x)*y-18*x*x+36*x-22},
		0.25*Vector{9*y*y+(18*x-9)*y+54*x*x-36*x+4,       18*x*y+9*x*x-9*x},
		0.25*Vector{9*y*y+(18*x-9)*y,                     54*y*y+(18*x-36)*y+9*x*x-9*x+4},
		0.25*Vector{(108*x-36)*y+162*x*x-180*x+36,        54*x*x-36*x},
		0.25*Vector{(18-108*x)*y-162*x*x+144*x-18,        18*x-54*x*x},
		0.25*Vector{(54*x+9)*y-27*y*y,                    -(54*x*y)+27*x*x+9*x},
		0.25*Vector{27*y*y+(9-54*x)*y,                    54*x*y-27*x*x+9*x},
		0.25*Vector{18*y-54*y*y,                          -(162*y*y)+(144-108*x)*y+18*x-18},
		0.25*Vector{54*y*y-36*y,                          162*y*y+(108*x-180)*y-36*x+36}
	};
}

///////////////////////////////////////////////////////////////////////////////
// Basis, 9 node cubic no11
///////////////////////////////////////////////////////////////////////////////
size_t TriangleCubicNo11Basis::size() const{
	return 9;
}

std::vector<Point> TriangleCubicNo11Basis::parametric_reference_points() const{
	return {
		{0, 0},
		{1, 0},
		{0, 1},
		{1.0/3.0, 0},
		{2.0/3.0, 0},
		{2.0/3.0, 1.0/3.0},
		{1.0/3.0, 2.0/3.0},
		{0, 2.0/3.0},
		{0, 1.0/3.0}
	};
}

std::vector<BasisType> TriangleCubicNo11Basis::basis_types() const{
	return std::vector<BasisType>(size(), BasisType::Nodal);
}

std::vector<double> TriangleCubicNo11Basis::value(Point xi) const{
	double x = xi.x();
	double y = xi.y();
	return {
		//
	};

}

std::vector<Vector> TriangleCubicNo11Basis::grad(Point xi) const{
	double x = xi.x();
	double y = xi.y();
	return {
		0.25*Vector{-(18*y*y)+(36-36*x)*y-54*x*x+72*x-22, -(54*y*y)+(72-36*x)*y-18*x*x+36*x-22},
		0.25*Vector{9*y*y+(18*x-9)*y+54*x*x-36*x+4,       18*x*y+9*x*x-9*x},
		0.25*Vector{9*y*y+(18*x-9)*y,                     54*y*y+(18*x-36)*y+9*x*x-9*x+4},
		0.25*Vector{(108*x-36)*y+162*x*x-180*x+36,        54*x*x-36*x},
		0.25*Vector{(18-108*x)*y-162*x*x+144*x-18,        18*x-54*x*x},
		0.25*Vector{(54*x+9)*y-27*y*y,                    -(54*x*y)+27*x*x+9*x},
		0.25*Vector{27*y*y+(9-54*x)*y,                    54*x*y-27*x*x+9*x},
		0.25*Vector{18*y-54*y*y,                          -(162*y*y)+(144-108*x)*y+18*x-18},
		0.25*Vector{54*y*y-36*y,                          162*y*y+(108*x-180)*y-36*x+36}
	};
}
///////////////////////////////////////////////////////////////////////////////
// Basis, hermite cubic
///////////////////////////////////////////////////////////////////////////////
TriangleHermiteBasis::TriangleHermiteBasis(std::shared_ptr<const IElementGeometry> geom): _geom(geom){
}

size_t TriangleHermiteBasis::size() const{
	return 10;
}

std::vector<Point> TriangleHermiteBasis::parametric_reference_points() const{
	return {
		{0, 0},
		{1, 0},
		{0, 1},
		{0, 0},
		{1, 0},
		{0, 1},
		{0, 0},
		{1, 0},
		{0, 1},
		{1.0/3.0, 1.0/3.0}
	};
}

std::vector<BasisType> TriangleHermiteBasis::basis_types() const{
	return {
		BasisType::Nodal,
		BasisType::Nodal,
		BasisType::Nodal,
		BasisType::Dx,
		BasisType::Dx,
		BasisType::Dx,
		BasisType::Dy,
		BasisType::Dy,
		BasisType::Dy,
		BasisType::Nodal
	};
}

std::vector<double> TriangleHermiteBasis::value(Point xi) const{
	double x = xi.x();
	double y = xi.y();
	JacobiMatrix jac = _geom->jacobi(xi);
	double C11 =  jac.j22 / jac.modj;
	double C12 = -jac.j21 / jac.modj;
	double C21 = -jac.j12 / jac.modj;
	double C22 =  jac.j11 / jac.modj;
	return {
		2*y*y*y+(13*x-3)*y*y+(13*x*x-13*x)*y+2*x*x*x-3*x*x+1,
		7*x*y*y+(7*x*x-7*x)*y-2*x*x*x+3*x*x,
		-(2*y*y*y)+(7*x+3)*y*y+(7*x*x-7*x)*y,
		(C21*(-y*y*y+(2-3*x)*y*y+(-(2*x*x)+3*x-1)*y)+C22*(2*x*y*y+(3*x*x-3*x)*y+x*x*x-2*x*x+x))*jac.modj,
		-((C22*(2*x*y*y+(2*x*x-2*x)*y-x*x*x+x*x)+C21*(x*y*y+(2*x*x-x)*y))*jac.modj),
		(C21*(-y*y*y+(2*x+1)*y*y+(2*x*x-2*x)*y)+C22*(2*x*y*y+(x*x-x)*y))*jac.modj,
		-((C11*(-y*y*y+(2-3*x)*y*y+(-(2*x*x)+3*x-1)*y)+C12*(2*x*y*y+(3*x*x-3*x)*y+x*x*x-2*x*x+x))*jac.modj),
		(C12*(2*x*y*y+(2*x*x-2*x)*y-x*x*x+x*x)+C11*(x*y*y+(2*x*x-x)*y))*jac.modj,
		-((C11*(-y*y*y+(2*x+1)*y*y+(2*x*x-2*x)*y)+C12*(2*x*y*y+(x*x-x)*y))*jac.modj),
		(27*x-27*x*x)*y-27*x*y*y
	};
}

std::vector<Vector> TriangleHermiteBasis::grad(Point xi) const{
	double x = xi.x();
	double y = xi.y();
	JacobiMatrix jac = _geom->jacobi(xi);
	double C11 =  jac.j22 / jac.modj;
	double C12 = -jac.j21 / jac.modj;
	double C21 = -jac.j12 / jac.modj;
	double C22 =  jac.j11 / jac.modj;
	return {
		Vector{13*y*y+(26*x-13)*y+6*x*x-6*x,                                          6*y*y+(26*x-6)*y+13*x*x-13*x},
		Vector{7*y*y+(14*x-7)*y-6*x*x+6*x,                                            14*x*y+7*x*x-7*x},
		Vector{7*y*y+(14*x-7)*y,                                                      -(6*y*y)+(14*x+6)*y+7*x*x-7*x},
		Vector{(C22*(2*y*y+(6*x-3)*y+3*x*x-4*x+1)+C21*((3-4*x)*y-3*y*y))*jac.modj,    (C21*(-(3*y*y)+(4-6*x)*y-2*x*x+3*x-1)+C22*(4*x*y+3*x*x-3*x))*jac.modj},
		Vector{-((C22*(2*y*y+(4*x-2)*y-3*x*x+2*x)+C21*(y*y+(4*x-1)*y))*jac.modj),     -((C22*(4*x*y+2*x*x-2*x)+C21*(2*x*y+2*x*x-x))*jac.modj)},
		Vector{(C21*(2*y*y+(4*x-2)*y)+C22*(2*y*y+(2*x-1)*y))*jac.modj,                (C21*(-(3*y*y)+(4*x+2)*y+2*x*x-2*x)+C22*(4*x*y+x*x-x))*jac.modj},
		Vector{-((C12*(2*y*y+(6*x-3)*y+3*x*x-4*x+1)+C11*((3-4*x)*y-3*y*y))*jac.modj), -((C11*(-(3*y*y)+(4-6*x)*y-2*x*x+3*x-1)+C12*(4*x*y+3*x*x-3*x))*jac.modj)},
		Vector{(C12*(2*y*y+(4*x-2)*y-3*x*x+2*x)+C11*(y*y+(4*x-1)*y))*jac.modj,        (C12*(4*x*y+2*x*x-2*x)+C11*(2*x*y+2*x*x-x))*jac.modj},
		Vector{-((C11*(2*y*y+(4*x-2)*y)+C12*(2*y*y+(2*x-1)*y))*jac.modj),             -((C11*(-(3*y*y)+(4*x+2)*y+2*x*x-2*x)+C12*(4*x*y+x*x-x))*jac.modj)},
		Vector{(27-54*x)*y-27*y*y,                                                    -(54*x*y)-27*x*x+27*x},
	};
}
