#ifndef __CFD_FEM_I_ELEMENT_HPP__
#define __CFD_FEM_I_ELEMENT_HPP__

#include "cfd/mat/densemat.hpp"
#include "cfd/grid/i_grid.hpp"
#include "cfd/geom/jacobi.hpp"
#include <functional>

namespace cfd{

///////////////////////////////////////////////////////////////////////////////
// Element Geometry
///////////////////////////////////////////////////////////////////////////////
class IElementGeometry{
public:
	virtual ~IElementGeometry() = default;

	virtual JacobiMatrix jacobi(Point xi) const = 0;
	virtual Point to_physical(Point xi) const  { _THROW_NOT_IMP_; }
	virtual Point to_parametric(Point p) const { _THROW_NOT_IMP_; }
	virtual Point parametric_center() const {_THROW_NOT_IMP_; }
};

///////////////////////////////////////////////////////////////////////////////
// Element Basis
///////////////////////////////////////////////////////////////////////////////
enum struct BasisType{
	Custom,
	Nodal,
	Dx,
	Dy,
	Dz
};

class IElementBasis{
public:
	virtual ~IElementBasis() = default;

	virtual size_t size() const = 0;
	virtual std::vector<Point> parametric_reference_points() const = 0;
	virtual std::vector<BasisType> basis_types() const = 0;
	virtual std::vector<double> value(Point xi) const = 0;
	virtual std::vector<Vector> grad(Point xi) const = 0;
	virtual std::vector<std::array<double, 6>> upper_hessian(Point xi) const { _THROW_NOT_IMP_; } 
};

///////////////////////////////////////////////////////////////////////////////
// Element Integrals
///////////////////////////////////////////////////////////////////////////////
class IElementIntegrals{
public:
	virtual ~IElementIntegrals() = default;

	// ⌠
	// ⎮ ϕᵢ dΩ
	// ⌡
	virtual std::vector<double> load_vector() const  { _THROW_NOT_IMP_; }
	// ⌠
	// ⎮ ϕⱼ ϕᵢ dΩ
	// ⌡
	virtual std::vector<double> mass_matrix() const  { _THROW_NOT_IMP_; }
	// ⌠
	// ⎮ ∇ϕⱼ⋅∇ϕᵢ dΩ
	// ⌡
	virtual std::vector<double> stiff_matrix() const { _THROW_NOT_IMP_; }
	// ⌠ 𝜕ϕⱼ
	// ⎮ ── ϕᵢ dΩ
	// ⌡ 𝜕x
	virtual std::vector<double> dx_matrix() const {_THROW_NOT_IMP_;}
	// ⌠ 𝜕ϕⱼ
	// ⎮ ── ϕᵢ dΩ
	// ⌡ 𝜕y
	virtual std::vector<double> dy_matrix() const {_THROW_NOT_IMP_;}
	// ⌠ 𝜕ϕⱼ
	// ⎮ ── ϕᵢ dΩ
	// ⌡ 𝜕z
	virtual std::vector<double> dz_matrix() const {_THROW_NOT_IMP_;}
	// ⌠
	// ⎮ u⃗⋅∇ϕⱼ ϕᵢ dΩ
	// ⌡
	virtual std::vector<double> transport_matrix(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }
	// ⌠
	// ⎮ ∇⋅u⃗ ϕᵢ dΩ
	// ⌡
	virtual std::vector<double> divergence_vector(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }
	// ⌠
	// ⎮ u⃗⋅∇ϕᵢ dΩ
	// ⌡
	virtual std::vector<double> divergence_vector_byparts(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }

	// ================================= stabilizators
	// ⌠ 
	// ⎮ ϕⱼ u⃗⋅∇ϕᵢ dΩ
	// ⌡
	virtual std::vector<double> mass_matrix_stab_supg(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }
	// ⌠ 
	// ⎮ u⃗⋅∇ϕᵢ dΩ
	// ⌡
	std::vector<double> load_vector_stab_supg(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { return divergence_vector_byparts(vx, vy, vz); }
	// ⌠
	// ⎮ ∇ϕⱼ⋅∇(u⃗⋅∇ϕᵢ) dΩ
	// ⌡
	virtual std::vector<double> stiff_matrix_stab_supg(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }
	// ⌠ 
	// ⎮ u⃗⋅∇ϕⱼ u⃗⋅∇ϕᵢ dΩ
	// ⌡
	virtual std::vector<double> transport_matrix_stab_supg(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }
	// ⌠ 𝜕ϕⱼ
	// ⎮ ── u⃗⋅∇ϕᵢ dΩ
	// ⌡ 𝜕x
	virtual std::vector<double> dx_matrix_stab_supg(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }
	// ⌠ 𝜕ϕⱼ
	// ⎮ ── u⃗⋅∇ϕᵢ dΩ
	// ⌡ 𝜕y
	virtual std::vector<double> dy_matrix_stab_supg(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }
	// ⌠ 𝜕ϕⱼ
	// ⎮ ── u⃗⋅∇ϕᵢ dΩ
	// ⌡ 𝜕z
	virtual std::vector<double> dz_matrix_stab_supg(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }
	// ⌠ 𝜕ϕⱼ
	// ⎮ ── ∇⋅(u⃗ ϕᵢ)dΩ
	// ⌡ 𝜕x
	virtual std::vector<double> dx_matrix_stab_supg2(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }
	// ⌠ 𝜕ϕⱼ
	// ⎮ ── ∇⋅(u⃗ ϕᵢ)dΩ
	// ⌡ 𝜕y
	virtual std::vector<double> dy_matrix_stab_supg2(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }
	// ⌠ 𝜕ϕⱼ
	// ⎮ ── ∇⋅(u⃗ ϕᵢ)dΩ
	// ⌡ 𝜕z
	virtual std::vector<double> dz_matrix_stab_supg2(
			const std::vector<double>& vx,
			const std::vector<double>& vy={},
			const std::vector<double>& vz={}) const { _THROW_NOT_IMP_; }


	// ⌠ 
	// ⎮ fᵢⱼ dΩ
	// ⌡
	class OperandArg;
	using OperandFunc = std::function<double(size_t, size_t, const OperandArg*)>;
	virtual std::vector<double> custom_matrix(OperandFunc f) const { _THROW_NOT_IMP_; }

};

class IElementIntegrals::OperandArg{
public:
	OperandArg(const IElementGeometry* geom, const IElementBasis* basis, Point xi);
	~OperandArg();

	const IElementGeometry* geom;
	const IElementBasis* basis;
	Point xi;

	double phi(size_t i) const;        // phi_i at (xi)
	Vector grad_phi(size_t i) const;   // dphi_i/dx,dy,dz at xi
	double laplace(size_t i) const;    // div(grad(phi_i)) at xi
	double modj() const;
	const JacobiMatrix* jacobi() const;
	double interpolate(const std::vector<double>& f) const;
	Vector interpolate(const std::vector<Vector>& f) const;
	double divergence(const std::vector<Vector>& f) const;
public:
	struct Cache;
	std::unique_ptr<Cache> _pcache;
};


///////////////////////////////////////////////////////////////////////////////
// FemElement
///////////////////////////////////////////////////////////////////////////////
struct FemElement{
	std::shared_ptr<const IElementGeometry> geometry;
	std::shared_ptr<const IElementBasis> basis;
	std::shared_ptr<const IElementIntegrals> integrals;
};

}
#endif
