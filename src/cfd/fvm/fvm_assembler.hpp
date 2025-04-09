#ifndef CFD_FVM_ASSEMBLER_HPP
#define CFD_FVM_ASSEMBLER_HPP

#include "cfd/mat/lodmat.hpp"
#include "cfd/grid/i_grid.hpp"

namespace cfd{

///////////////////////////////////////////////////////////////////////////////
// FvmExtendedCollocations
///////////////////////////////////////////////////////////////////////////////

struct FvmExtendedCollocations{
public:
	explicit FvmExtendedCollocations(const IGrid& grid);

	std::vector<Point> points;
	std::vector<size_t> cell_collocations;
	std::vector<size_t> face_collocations;

	/// number of collocations
	size_t size() const;

	/// index of a cell for the given collocation. Throws if icolloc is not a cell collocation
	size_t cell_index(size_t icolloc) const;

	/// index of a face for the given collocation. Throws if icolloc is not a face collocation
	size_t face_index(size_t icolloc) const;
	
	std::array<size_t, 2> tab_face_colloc(size_t iface) const;

	std::vector<size_t> tab_colloc_colloc(size_t icolloc) const;

	bool is_boundary_colloc(size_t icolloc) const;
	bool is_internal_colloc(size_t icolloc) const;
private:
	std::vector<std::array<size_t, 2>> _tab_face_colloc;
	std::vector<std::vector<size_t>> _tab_colloc_colloc;
	std::vector<size_t> _face_indices;
};

///////////////////////////////////////////////////////////////////////////////
// FvmCellGradient
///////////////////////////////////////////////////////////////////////////////

struct IFvmCellGradient{
	virtual ~IFvmCellGradient() = default;

	std::vector<Vector> compute(const std::vector<double>& u) const;
	std::vector<Vector> compute(const double* u) const;
protected:
	std::array<CsrMatrix, 3> _data;
};

struct LeastSquaresFvmCellGradient: public IFvmCellGradient{
	LeastSquaresFvmCellGradient(const IGrid& grid, const FvmExtendedCollocations& colloc);
};

struct GaussLinearFvmCellGradient: public IFvmCellGradient{
	GaussLinearFvmCellGradient(const IGrid& grid, const FvmExtendedCollocations& colloc);
};

using FvmCellGradient = LeastSquaresFvmCellGradient;

///////////////////////////////////////////////////////////////////////////////
// DfDn on faces
///////////////////////////////////////////////////////////////////////////////

/// DfDn on faces computer
struct FvmFacesDn{
	explicit FvmFacesDn(const IGrid& grid);
	FvmFacesDn(const IGrid& grid, const FvmExtendedCollocations& colloc);

	/// computes dfdn for each grid face
	std::vector<double> compute(const std::vector<double>& f) const;
	std::vector<double> compute(const double* f) const;

	/// computes dfdn for the given grid face
	double compute(size_t iface, const std::vector<double>& f) const;
	double compute(size_t iface, const double* f) const;

	/// returns dfdn as a linear combination of collocation values
	const std::map<size_t, double>& linear_combination(size_t iface) const;
private:
	LodMatrix _dfdn;
};

/// DfDn on faces computer
struct FvmLinformFacesDn{
	using linform_t = std::vector<std::pair<size_t, double>>;

	explicit FvmLinformFacesDn(const IGrid& grid);
	FvmLinformFacesDn(const IGrid& grid, const FvmExtendedCollocations& colloc);

	/// returns dfdn as a linear combination of collocation values
	const linform_t& linear_combination(size_t iface) const;
private:
	std::vector<linform_t> _linform;
};

}
#endif
