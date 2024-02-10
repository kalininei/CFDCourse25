#ifndef MAT_CSRMAT_HPP
#define MAT_CSRMAT_HPP

#include "cfd/mat/i_sparse_mat.hpp"

namespace cfd{

/**
 * @brief 'Compressed sparsed row' stencil
 */
class CsrStencil: public ISparseMatrix{
public:
	virtual ~CsrStencil() = default;
	/**
	 * @brief Fills CSR stencil by address and column vectors
	 *
	 * @param addr  address vector
	 * @param cols  column vector
	 */
	void set_stencil(std::vector<size_t>&& addr, std::vector<size_t>&& cols);
	void set_stencil(const std::vector<std::set<size_t>>& stencil_set);

	/**
	 * @brief Address array getter
	 */
	const std::vector<size_t>& addr() const;

	/**
	 * @brief Columns array getter
	 */
	const std::vector<size_t>& cols() const;

	/**
	 * @brief Consistency check. Throws if stencil structure is invalid
	 */
	virtual void validate() const;

	size_t get_address(size_t irow, size_t icol) const;

	// overrides
	size_t n_rows() const override;
	size_t n_nonzeros() const override;
	bool is_in_stencil(size_t irow, size_t icol) const override;
	double value(size_t irow, size_t icol) const override;
	std::vector<double> mult_vec(const std::vector<double>& u) const override;
	double mult_vec(size_t irow, const std::vector<double>& u) const override;
private:
	std::vector<size_t> _addr = {0};
	std::vector<size_t> _cols;
};

/**
 * @brief 'Compressed sparsed row' matrix format
 */
class CsrMatrix: public CsrStencil{
public:
	CsrMatrix() = default;
	CsrMatrix(const CsrStencil& stencil): CsrStencil(stencil), _vals(stencil.n_nonzeros(), 0.0){}

	/**
	 * @brief Set matrix values
	 * 
	 * @param vals  values array
	 */
	void set_values(std::vector<double>&& vals);

	/**
	 * @brief Values array getter
	 */
	const std::vector<double>& vals() const;
	std::vector<double>& vals();

	void set_unit_row(size_t irow);

	// overrides
	void validate() const override;
	double value(size_t irow, size_t icol) const override;
	std::vector<double> mult_vec(const std::vector<double>& u) const override;
	double mult_vec(size_t irow, const std::vector<double>& u) const override;
private:
	std::vector<double> _vals;
};


}

#endif
