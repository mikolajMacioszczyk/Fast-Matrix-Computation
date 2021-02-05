#pragma once
#include "RandomFloatGenerator.h"

namespace MyAlgebra
{
	class Matrix
	{
	public:
		// precision according to which equality of matrices is checked
		static const float ALG_PRECISION;

		// =========================================================================
		// CONSTRUCTORS:
		// =========================================================================

		// Creates matrix with the posibility of random initialization
		Matrix(unsigned int row_cnt, unsigned int col_cnt, bool rand_init = false);

		// Creates squared diagonal matrix
		Matrix(unsigned int row_cnt, float diagonal);

		Matrix(const Matrix& rhs);

		Matrix(Matrix&& rhs);

		// =========================================================================
		// DESTRUCTOR:
		// =========================================================================
		~Matrix();


		// =========================================================================
		// ASSIGEMENT OPERATORS:
		// =========================================================================

		const Matrix& operator=(const Matrix& rhs);

		// Converting a matrix to a diagonal matrix
		const Matrix& operator=(float diagonal);

		const Matrix& operator=(Matrix&& rhs);


		// =========================================================================
		// ARRAY OPERATOR
		// =========================================================================

		float* operator[](unsigned int row_ind);

		// =========================================================================
		// ALEGEBRAIC OPERATORS
		// =========================================================================

		Matrix operator*(const Matrix& rhs) const;

		Matrix operator*(float multiplier) const;

		Matrix operator+(const Matrix& rhs) const;
		Matrix operator+(Matrix&& rhs) const;

		Matrix operator-(const Matrix& rhs) const;
		Matrix operator-(Matrix&& rhs) const;

		// Unary minus - changes the sign of all cells
		Matrix operator-() const;

		// Transpose matrix
		Matrix operator~() const;

		// Power matrix
		Matrix operator^(int power) const;

		// Comparison of matrices with constant ALG_PRECISION
		bool operator==(const Matrix& rhs) const;

	private:
		float* data;
		unsigned int row;
		unsigned int coll;
		unsigned int fullSize;

		static RandomFloatGenerator GENERATOR;

		// =========================================================================
		// CREATION / DESTRUCTION HELPER METHODS
		// =========================================================================
		void CreateData(bool randomInit);
		void CopyData(const Matrix& rhs);
		void FillDiagonal(float diagonal);
		void MoveData(Matrix& rhs);
		void ClearData();

		// =========================================================================
		// ALGEBRAIC HELPER METHODS
		// =========================================================================
		void operatorPlusHelper(const Matrix* source, Matrix* destination, int startIdx, int endIdx) const;
		void operatorPlusHelperMove(Matrix* destination, int startIdx, int endIdx) const;
		void operatorMinusHelper(const Matrix* source, Matrix* destination, int startIdx, int endIdx) const;

		void operatorMultiplyMatrixHelper1(const Matrix* source, Matrix* destination, unsigned int endRow, unsigned int endCol) const;
		void operatorMultiplyMatrixHelper2(const Matrix* source, Matrix* destination, unsigned int endRow, unsigned int endCol) const;
		void operatorMultiplyMatrixHelper3(const Matrix* source, Matrix* destination, unsigned int endRow, unsigned int endCol) const;
		void operatorMultiplyMatrixHelper4(const Matrix* source, Matrix* destination, unsigned int endRow, unsigned int endCol) const;

		void transposeMatrixHelper(Matrix* destination, int startRow, int endRow) const;
	};


	Matrix operator*(float multiplier, const MyAlgebra::Matrix& rhs);
}