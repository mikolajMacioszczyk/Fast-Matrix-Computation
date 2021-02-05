#include "Matrix.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <immintrin.h>
#include "MatrixException.h"
#include "RandomFloatGenerator.h"

const float MyAlgebra::Matrix::ALG_PRECISION = 0.00001;
MyAlgebra::RandomFloatGenerator MyAlgebra::Matrix::GENERATOR = RandomFloatGenerator();

#define DEFAULT_SIZE 0
#define DEFAULT_VALUE 0
#define BLOCK_SIZE 32
#define VECTORIZATION_SIZE 8

MyAlgebra::Matrix::Matrix(unsigned int row, unsigned int col, bool randInit)
	: row(row), coll(col), data(nullptr), fullSize(row* col)
{
	CreateData(randInit);
}

MyAlgebra::Matrix::Matrix(unsigned int row, float diagonal)
	: row(row), coll(row), data(nullptr), fullSize(row* row)
{
	CreateData(false);
	for (unsigned int i = 0; i < fullSize; i += coll + 1)
	{
		data[i] = diagonal;
	}
}

MyAlgebra::Matrix::Matrix(const Matrix& rhs)
	: row(-1), coll(-1), data(nullptr), fullSize(-1)
{
	CopyData(rhs);
}

MyAlgebra::Matrix::Matrix(Matrix&& rhs)
	: row(rhs.row), coll(rhs.coll), data(nullptr), fullSize(rhs.fullSize)
{
	MoveData(rhs);
}

MyAlgebra::Matrix::~Matrix()
{
	ClearData();
}

const MyAlgebra::Matrix& MyAlgebra::Matrix::operator=(const Matrix& rhs)
{
	CopyData(rhs);
	return *this;
}

const MyAlgebra::Matrix& MyAlgebra::Matrix::operator=(float diagonal)
{
	for (unsigned int i = 0; i < fullSize; ++i)
	{
		data[i] = DEFAULT_VALUE;
	}
	for (unsigned int i = 0; i < fullSize; i += coll + 1)
	{
		data[i] = diagonal;
	}
	return *this;
}

const MyAlgebra::Matrix& MyAlgebra::Matrix::operator=(Matrix&& rhs)
{
	MoveData(rhs);
	return *this;
}

float* MyAlgebra::Matrix::operator[](unsigned int idx)
{
	if (idx < DEFAULT_VALUE || idx >= row)
	{
		throw MatrixException("operator[]", "Index " + std::to_string(idx) + " out of bounds");
	}
	return data + ((size_t)idx) * coll;
}


inline void MultiplyBlock(float* destinationPtr, float* lhsPtr, float* rhsPtr, unsigned int destinationColl, unsigned int dataColl,
	unsigned int sourceColl, unsigned int blockRowIdx, unsigned int startRow, unsigned int blockColumnIdx, unsigned int startColl, unsigned int offset)
{
	for (unsigned int i = startRow + (blockRowIdx * BLOCK_SIZE); i < startRow + (blockRowIdx * BLOCK_SIZE) + BLOCK_SIZE; ++i)
	{
		float* destinationData = destinationPtr + ((size_t)i) * destinationColl;
		for (unsigned int k = offset; k < offset + BLOCK_SIZE; ++k)
		{

			const float lhsDataIK = lhsPtr[i * dataColl + k];
			const float* rhsData = rhsPtr + ((size_t)k) * sourceColl;
			__m256 memOriginal, memSource1, memSource2, memOperational, memOperational2;
			memSource1 = _mm256_set1_ps(lhsDataIK);

			for (unsigned int j = startColl + (blockColumnIdx * BLOCK_SIZE); j < startColl + (blockColumnIdx * BLOCK_SIZE) + BLOCK_SIZE; j += VECTORIZATION_SIZE)
			{
				memSource2 = _mm256_load_ps(rhsData + j);
				memOriginal = _mm256_load_ps(destinationData + j);

				memOperational = _mm256_mul_ps(memSource1, memSource2);
				memOperational2 = _mm256_add_ps(memOperational, memOriginal);

				_mm256_store_ps(destinationData + j, memOperational2);
			}
		}
	}
}

void MyAlgebra::Matrix::operatorMultiplyMatrixHelper1(const Matrix* source, Matrix* destination, unsigned int rowSize, unsigned int colSize) const
{
	int endRow = rowSize / 2;
	int endCol = colSize / 2;
	const int blockRowCount = endRow / BLOCK_SIZE;
	const int blockColumnCount = endCol / BLOCK_SIZE;

	for (int blockRowIdx = 0; blockRowIdx < blockRowCount; ++blockRowIdx)
	{
		for (int blockColumnIdx = 0; blockColumnIdx < blockColumnCount; blockColumnIdx++)
		{
			for (int offset = 0; offset < coll / BLOCK_SIZE * BLOCK_SIZE; offset += BLOCK_SIZE)
			{
				MultiplyBlock(destination->data, data, source->data, destination->coll, coll, source->coll, blockRowIdx, DEFAULT_VALUE, blockColumnIdx, DEFAULT_VALUE, offset);
			}
		}
	}

	const unsigned int remainsStart = coll / BLOCK_SIZE * BLOCK_SIZE;
	for (unsigned int i = 0; i < endRow; ++i)
	{
		register float* destinationData = destination->data + ((size_t)i) * destination->coll;
		for (unsigned int k = remainsStart; k < coll; ++k)
		{
			const float dataIK = data[i * coll + k];
			for (unsigned int j = 0; j < endCol; ++j)
			{
				destinationData[j] += dataIK * source->data[k * source->coll + j];
			}
		}
	}

	for (unsigned int k = 0; k < coll; ++k)
	{
		for (unsigned int i = 0; i < row; ++i)
		{
			float* destinationData = destination->data + ((size_t)i) * destination->coll;
			const float dataIK = data[i * coll + k];
			for (unsigned int j = colSize; j < source->coll; ++j)
			{
				destinationData[j] += dataIK * source->data[k * source->coll + j];
			}
		}
		for (unsigned int i = rowSize; i < row; ++i)
		{
			float* destinationData = destination->data + ((size_t)i) * destination->coll;
			const register float dataIK = data[i * coll + k];
			for (unsigned int j = 0; j < colSize; ++j)
			{
				destinationData[j] += dataIK * source->data[k * source->coll + j];
			}
		}
	}
}

void MyAlgebra::Matrix::operatorMultiplyMatrixHelper2(const Matrix* source, Matrix* destination, unsigned int rowSize, unsigned int colSize) const
{
	int startRow = rowSize / 2;
	int endCol = colSize / 2;
	const int blockRowCount = (rowSize - startRow) / BLOCK_SIZE;
	const int blockColumnCount = (endCol) / BLOCK_SIZE;

	for (int blockRowIdx = 0; blockRowIdx < blockRowCount; ++blockRowIdx)
	{
		for (int blockColumnIdx = 0; blockColumnIdx < blockColumnCount; blockColumnIdx++)
		{
			for (int offset = 0; offset < coll / BLOCK_SIZE * BLOCK_SIZE; offset += BLOCK_SIZE)
			{
				MultiplyBlock(destination->data, data, source->data, destination->coll, coll, source->coll, blockRowIdx, startRow, blockColumnIdx, DEFAULT_VALUE, offset);
			}
		}
	}

	const unsigned int remainsStart = coll / BLOCK_SIZE * BLOCK_SIZE;
	for (unsigned int i = startRow; i < rowSize; ++i)
	{
		register float* destinationData = destination->data + ((size_t)i) * destination->coll;
		for (unsigned int k = remainsStart; k < coll; ++k)
		{
			const float dataIK = data[i * coll + k];
			for (unsigned int j = 0; j < endCol; ++j)
			{
				destinationData[j] += dataIK * source->data[k * source->coll + j];
			}
		}
	}
}

void MyAlgebra::Matrix::operatorMultiplyMatrixHelper3(const Matrix* source, Matrix* destination, unsigned int rowSize, unsigned int colSize) const
{
	const int endRow = rowSize / 2;
	const int startCol = colSize / 2;
	const int blockRowCount = (endRow) / BLOCK_SIZE;
	const int blockColumnCount = (colSize - startCol) / BLOCK_SIZE;

	for (int blockRowIdx = 0; blockRowIdx < blockRowCount; ++blockRowIdx)
	{
		for (int blockColumnIdx = 0; blockColumnIdx < blockColumnCount; blockColumnIdx++)
		{
			for (int offset = 0; offset < coll / BLOCK_SIZE * BLOCK_SIZE; offset += BLOCK_SIZE)
			{
				MultiplyBlock(destination->data, data, source->data, destination->coll, coll, source->coll, blockRowIdx, DEFAULT_VALUE, blockColumnIdx, startCol, offset);
			}
		}
	}

	const unsigned int remainsStart = coll / BLOCK_SIZE * BLOCK_SIZE;
	for (unsigned int i = 0; i < endRow; ++i)
	{
		register float* destinationData = destination->data + ((size_t)i) * destination->coll;
		for (unsigned int k = remainsStart; k < coll; ++k)
		{
			const register float dataIK = data[i * coll + k];
			for (unsigned int j = startCol; j < colSize; ++j)
			{
				destinationData[j] += dataIK * source->data[k * source->coll + j];
			}
		}
	}
}

void MyAlgebra::Matrix::operatorMultiplyMatrixHelper4(const Matrix* source, Matrix* destination, unsigned int rowSize, unsigned int colSize) const
{
	const int startRow = rowSize / 2;
	const int startCol = colSize / 2;
	const int blockRowCount = (rowSize - startRow) / BLOCK_SIZE;
	const int blockColumnCount = (colSize - startCol) / BLOCK_SIZE;

	for (int blockRowIdx = 0; blockRowIdx < blockRowCount; ++blockRowIdx)
	{
		for (int blockColumnIdx = 0; blockColumnIdx < blockColumnCount; blockColumnIdx++)
		{
			for (int offset = 0; offset < coll / BLOCK_SIZE * BLOCK_SIZE; offset += BLOCK_SIZE)
			{
				MultiplyBlock(destination->data, data, source->data, destination->coll, coll, source->coll, blockRowIdx, startRow, blockColumnIdx, startCol, offset);
			}
		}
	}

	const unsigned int remainsStart = coll / BLOCK_SIZE * BLOCK_SIZE;
	for (unsigned int i = startRow; i < rowSize; ++i)
	{
		register float* destinationData = destination->data + ((size_t)i) * destination->coll;
		for (unsigned int k = remainsStart; k < coll; ++k)
		{
			const register float dataIK = data[i * coll + k];
			for (unsigned int j = startCol; j < colSize; ++j)
			{
				destinationData[j] += dataIK * source->data[k * source->coll + j];
			}
		}
	}
}

MyAlgebra::Matrix MyAlgebra::Matrix::operator*(const Matrix& rhs) const
{
	if (coll != rhs.row)
	{
		throw MatrixException("operator*", "Rhs row " + std::to_string(rhs.row) + " should be equal to this coll " + std::to_string(coll));
	}
	Matrix result(row, rhs.coll);
	if (row * rhs.coll < 260000)
	{
		const int canBeUnfolded = (rhs.coll / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
		for (int i = 0; i < row; ++i)
		{
			register float* resultPtr = result.data + ((size_t)i) * result.coll;
			for (int k = 0; k < coll; ++k)
			{
				const register float dataIK = data[i * coll + k];
				const register float* rhsPtr = rhs.data + ((size_t)k) * rhs.coll;

				__m256 memResult, memData, memRhs, memOperational;
				memData = _mm256_set1_ps(dataIK);
				for (int j = 0; j < canBeUnfolded; j += VECTORIZATION_SIZE)
				{
					memRhs = _mm256_load_ps(rhsPtr + j);
					memResult = _mm256_load_ps(resultPtr + j);

					memOperational = _mm256_mul_ps(memData, memRhs);
					memOperational = _mm256_add_ps(memResult, memOperational);

					_mm256_store_ps(resultPtr + j, memOperational);
				}
				for (int j = canBeUnfolded; j < rhs.coll; ++j)
				{
					result.data[i * result.coll + j] += data[i * coll + k] * rhs.data[k * rhs.coll + j];
				}
			}
		}
	}
	else
	{
		const unsigned int rowBlockedPart = (row / (2 * BLOCK_SIZE)) * 2 * BLOCK_SIZE;
		const unsigned int columnBlockedPart = (rhs.coll / (2 * BLOCK_SIZE)) * 2 * BLOCK_SIZE;
		std::thread helper1(&Matrix::operatorMultiplyMatrixHelper1, this, &rhs, &result, rowBlockedPart, columnBlockedPart);
		std::thread helper2(&Matrix::operatorMultiplyMatrixHelper2, this, &rhs, &result, rowBlockedPart, columnBlockedPart);
		std::thread helper3(&Matrix::operatorMultiplyMatrixHelper3, this, &rhs, &result, rowBlockedPart, columnBlockedPart);
		operatorMultiplyMatrixHelper4(&rhs, &result, rowBlockedPart, columnBlockedPart);
		helper2.join();
		helper3.join();
		helper1.join();
	}
	return result;
}

MyAlgebra::Matrix MyAlgebra::Matrix::operator*(float multiplier) const
{
	MyAlgebra::Matrix result(row, coll);
	const unsigned int canBeUnfolded = (fullSize / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
	for (unsigned int j = 0; j < canBeUnfolded; j += VECTORIZATION_SIZE)
	{
		__m256 memData, memMult, memOperational;

		memData = _mm256_load_ps(data + j);
		memMult = _mm256_set1_ps(multiplier);

		memOperational = _mm256_mul_ps(memData, memMult);

		_mm256_store_ps(result.data + j, memOperational);
	}
	for (unsigned int j = canBeUnfolded; j < fullSize; ++j)
	{
		result.data[j] = multiplier * data[j];
	}
	return result;
}

void MyAlgebra::Matrix::operatorPlusHelper(const Matrix* source, Matrix* destination, int startIdx, int endIdx) const
{
	const unsigned int canBeUnfloaded = startIdx + ((endIdx - startIdx) / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
	for (unsigned int j = startIdx; j < canBeUnfloaded; j += VECTORIZATION_SIZE)
	{
		destination->data[j] = data[j] + source->data[j];
		destination->data[j + 1] = data[j + 1] + source->data[j + 1];
		destination->data[j + 2] = data[j + 2] + source->data[j + 2];
		destination->data[j + 3] = data[j + 3] + source->data[j + 3];
		destination->data[j + 4] = data[j + 4] + source->data[j + 4];
		destination->data[j + 5] = data[j + 5] + source->data[j + 5];
		destination->data[j + 6] = data[j + 6] + source->data[j + 6];
		destination->data[j + 7] = data[j + 7] + source->data[j + 7];
	}
	for (unsigned int j = canBeUnfloaded; j < endIdx; ++j)
	{
		destination->data[j] = data[j] + source->data[j];
	}
}

void MyAlgebra::Matrix::operatorPlusHelperMove(Matrix* destination, int startIdx, int endIdx) const
{
	const int canBeUnfloaded = startIdx + ((endIdx - startIdx) / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
	for (int j = startIdx; j < canBeUnfloaded; j += VECTORIZATION_SIZE)
	{
		destination->data[j] += data[j];
		destination->data[j + 1] += data[j + 1];
		destination->data[j + 2] += data[j + 2];
		destination->data[j + 3] += data[j + 3];
		destination->data[j + 4] += data[j + 4];
		destination->data[j + 5] += data[j + 5];
		destination->data[j + 6] += data[j + 6];
		destination->data[j + 7] += data[j + 7];
	}
	for (int j = canBeUnfloaded; j < endIdx; ++j)
	{
		destination->data[j] += data[j];
	}
}

MyAlgebra::Matrix MyAlgebra::Matrix::operator+(const Matrix& rhs) const
{
	if (row != rhs.row || coll != rhs.coll)
	{
		throw MatrixException("operator+", "Rhs row " + std::to_string(rhs.row) + " should be equal to this row " + std::to_string(row) +
			", Rhs coll " + std::to_string(rhs.coll) + " should be equal to this coll " + std::to_string(coll));
	}
	MyAlgebra::Matrix result(row, coll);
	if (row * coll < 5000000)
	{
		const unsigned int canBeUnfloaded = (fullSize / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
		for (unsigned int j = 0; j < canBeUnfloaded; j += VECTORIZATION_SIZE)
		{
			result.data[j] = data[j] + rhs.data[j];
			result.data[j + 1] = data[j + 1] + rhs.data[j + 1];
			result.data[j + 2] = data[j + 2] + rhs.data[j + 2];
			result.data[j + 3] = data[j + 3] + rhs.data[j + 3];
			result.data[j + 4] = data[j + 4] + rhs.data[j + 4];
			result.data[j + 5] = data[j + 5] + rhs.data[j + 5];
			result.data[j + 6] = data[j + 6] + rhs.data[j + 6];
			result.data[j + 7] = data[j + 7] + rhs.data[j + 7];
		}
		for (unsigned int j = canBeUnfloaded; j < fullSize; ++j)
		{
			result.data[j] = data[j] + rhs.data[j];
		}
	}
	else
	{
		std::thread helper1(&Matrix::operatorPlusHelper, this, &rhs, &result, 0, fullSize / 2);
		operatorPlusHelper(&rhs, &result, fullSize / 2, fullSize);
		helper1.join();
	}
	return result;
}

MyAlgebra::Matrix MyAlgebra::Matrix::operator+(Matrix&& rhs) const
{
	if (row != rhs.row || coll != rhs.coll)
	{
		throw MatrixException("operator+", "Rhs row " + std::to_string(rhs.row) + " should be equal to this row " + std::to_string(row) +
			", Rhs coll " + std::to_string(rhs.coll) + " should be equal to this coll " + std::to_string(coll));
	}
	if (row * coll < 5000000)
	{
		const int canBeUnfloaded = (fullSize / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
		for (int j = 0; j < canBeUnfloaded; j += VECTORIZATION_SIZE)
		{
			rhs.data[j] += data[j];
			rhs.data[j + 1] += data[j + 1];
			rhs.data[j + 2] += data[j + 2];
			rhs.data[j + 3] += data[j + 3];
			rhs.data[j + 4] += data[j + 4];
			rhs.data[j + 5] += data[j + 5];
			rhs.data[j + 6] += data[j + 6];
			rhs.data[j + 7] += data[j + 7];
		}
		for (int j = canBeUnfloaded; j < fullSize; ++j)
		{
			rhs.data[j] += data[j];
		}
	}
	else
	{
		std::thread helper1(&Matrix::operatorPlusHelperMove, this, &rhs, 0, fullSize / 2);
		operatorPlusHelperMove(&rhs, fullSize / 2, fullSize);
		helper1.join();
	}
	return std::move(rhs);
}

void MyAlgebra::Matrix::operatorMinusHelper(const Matrix* source, Matrix* destination, int startIdx, int endIdx) const
{
	const unsigned int canBeUnfloaded = startIdx + ((endIdx - startIdx) / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
	for (unsigned int j = startIdx; j < canBeUnfloaded; j += VECTORIZATION_SIZE)
	{
		destination->data[j] = data[j] - source->data[j];
		destination->data[j + 1] = data[j + 1] - source->data[j + 1];
		destination->data[j + 2] = data[j + 2] - source->data[j + 2];
		destination->data[j + 3] = data[j + 3] - source->data[j + 3];
		destination->data[j + 4] = data[j + 4] - source->data[j + 4];
		destination->data[j + 5] = data[j + 5] - source->data[j + 5];
		destination->data[j + 6] = data[j + 6] - source->data[j + 6];
		destination->data[j + 7] = data[j + 7] - source->data[j + 7];
	}
	for (unsigned int j = canBeUnfloaded; j < endIdx; ++j)
	{
		destination->data[j] = data[j] - source->data[j];
	}
}

MyAlgebra::Matrix MyAlgebra::Matrix::operator-(const Matrix& rhs) const
{
	if (row != rhs.row || coll != rhs.coll)
	{
		throw MatrixException("operator-", "Rhs row " + std::to_string(rhs.row) + " should be equal to this row " + std::to_string(row) +
			", Rhs coll " + std::to_string(rhs.coll) + " should be equal to this coll " + std::to_string(coll));
	}
	MyAlgebra::Matrix result(row, coll);
	if (row * coll < 5000000)
	{
		const unsigned int canBeUnfloaded = (fullSize / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
		for (unsigned int j = 0; j < canBeUnfloaded; j += VECTORIZATION_SIZE)
		{
			result.data[j] = data[j] - rhs.data[j];
			result.data[j + 1] = data[j + 1] - rhs.data[j + 1];
			result.data[j + 2] = data[j + 2] - rhs.data[j + 2];
			result.data[j + 3] = data[j + 3] - rhs.data[j + 3];
			result.data[j + 4] = data[j + 4] - rhs.data[j + 4];
			result.data[j + 5] = data[j + 5] - rhs.data[j + 5];
			result.data[j + 6] = data[j + 6] - rhs.data[j + 6];
			result.data[j + 7] = data[j + 7] - rhs.data[j + 7];
		}
		for (unsigned int j = canBeUnfloaded; j < fullSize; ++j)
		{
			result.data[j] = data[j] - rhs.data[j];
		}
	}
	else
	{
		std::thread helper1(&Matrix::operatorMinusHelper, this, &rhs, &result, 0, fullSize / 2);
		operatorMinusHelper(&rhs, &result, fullSize / 2, fullSize);
		helper1.join();
	}
	return result;
}

MyAlgebra::Matrix MyAlgebra::Matrix::operator-(Matrix&& rhs) const
{
	if (row != rhs.row || coll != rhs.coll)
	{
		throw MatrixException("operator-", "Rhs row " + std::to_string(rhs.row) + " should be equal to this row " + std::to_string(row) +
			", Rhs coll " + std::to_string(rhs.coll) + " should be equal to this coll " + std::to_string(coll));
	}
	if (row * coll < 5000000)
	{
		const int canBeUnfloaded = (fullSize / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
		for (int j = 0; j < canBeUnfloaded; j += VECTORIZATION_SIZE)
		{
			rhs.data[j] = data[j] - rhs.data[j];
			rhs.data[j + 1] = data[j + 1] - rhs.data[j + 1];
			rhs.data[j + 2] = data[j + 2] - rhs.data[j + 2];
			rhs.data[j + 3] = data[j + 3] - rhs.data[j + 3];
			rhs.data[j + 4] = data[j + 4] - rhs.data[j + 4];
			rhs.data[j + 5] = data[j + 5] - rhs.data[j + 5];
			rhs.data[j + 6] = data[j + 6] - rhs.data[j + 6];
			rhs.data[j + 7] = data[j + 7] - rhs.data[j + 7];
		}
		for (int j = canBeUnfloaded; j < fullSize; ++j)
		{
			rhs.data[j] = data[j] - rhs.data[j];
		}
	}
	else
	{
		std::thread helper1(&Matrix::operatorMinusHelper, this, &rhs, &rhs, 0, fullSize / 2);
		operatorMinusHelper(&rhs, &rhs, fullSize / 2, fullSize);
		helper1.join();
	}
	return std::move(rhs);
}

MyAlgebra::Matrix MyAlgebra::Matrix::operator-() const
{
	MyAlgebra::Matrix result(row, coll);
	const unsigned int canBeFolded = (fullSize / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
	for (unsigned int i = 0; i < canBeFolded; i += VECTORIZATION_SIZE)
	{
		__m256 memData, memZeros, memResult;
		memData = _mm256_load_ps(data + i);
		memZeros = _mm256_setzero_ps();
		memResult = _mm256_sub_ps(memZeros, memData);
		_mm256_store_ps(result.data + i, memResult);
	}
	for (unsigned int j = canBeFolded; j < fullSize; ++j)
	{
		result.data[j] = -data[j];
	}
	return result;
}

void MyAlgebra::Matrix::transposeMatrixHelper(Matrix* destination, int startRow, int endRow) const
{
	const unsigned int canBeUnfloaded = (coll / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
	for (unsigned int i = startRow; i < endRow; ++i)
	{
		for (unsigned int j = 0; j < canBeUnfloaded; j += VECTORIZATION_SIZE)
		{
			destination->data[j * row + i] = data[i * coll + j];
			destination->data[(j + 1) * row + i] = data[i * coll + j + 1];
			destination->data[(j + 2) * row + i] = data[i * coll + j + 2];
			destination->data[(j + 3) * row + i] = data[i * coll + j + 3];
			destination->data[(j + 4) * row + i] = data[i * coll + j + 4];
			destination->data[(j + 5) * row + i] = data[i * coll + j + 5];
			destination->data[(j + 6) * row + i] = data[i * coll + j + 6];
			destination->data[(j + 7) * row + i] = data[i * coll + j + 7];
		}
		for (unsigned int j = canBeUnfloaded; j < coll; ++j)
		{
			destination->data[j * row + i] = data[i * coll + j];
		}
	}
}

MyAlgebra::Matrix MyAlgebra::Matrix::operator~() const
{
	MyAlgebra::Matrix result(coll, row);
	if (row * coll < 2500000)
	{
		const unsigned int canBeFolded = (coll / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
		for (unsigned int i = 0; i < row; ++i)
		{
			for (unsigned int j = 0; j < canBeFolded; j += VECTORIZATION_SIZE)
			{
				result.data[j * row + i] = data[i * coll + j];
				result.data[(j + 1) * row + i] = data[i * coll + j + 1];
				result.data[(j + 2) * row + i] = data[i * coll + j + 2];
				result.data[(j + 3) * row + i] = data[i * coll + j + 3];
				result.data[(j + 4) * row + i] = data[i * coll + j + 4];
				result.data[(j + 5) * row + i] = data[i * coll + j + 5];
				result.data[(j + 6) * row + i] = data[i * coll + j + 6];
				result.data[(j + 7) * row + i] = data[i * coll + j + 7];
			}
			for (unsigned int j = canBeFolded; j < coll; ++j)
			{
				result.data[j * row + i] = data[i * coll + j];
			}
		}
	}
	else
	{
		std::thread helper1(&Matrix::transposeMatrixHelper, this, &result, 0, row / 4);
		std::thread helper2(&Matrix::transposeMatrixHelper, this, &result, row / 4, 2 * row / 4);
		std::thread helper3(&Matrix::transposeMatrixHelper, this, &result, 2 * row / 4, 3 * row / 4);
		transposeMatrixHelper(&result, 3 * row / 4, row);
		helper1.join();
		helper2.join();
		helper3.join();
	}
	return result;
}

MyAlgebra::Matrix MyAlgebra::Matrix::operator^(int power) const
{
	if (power < 0 || row != coll)
	{
		throw MatrixException("operator^", "Row " + std::to_string(row) + " should be equal to " + std::to_string(coll));
	}
	if (power == 0)
	{
		return Matrix(row, 1.0f);
	}
	Matrix result = *this;
	for (int i = 1; i < power; i++)
	{
		result = std::move(result * (*this));
	}
	return result;
}

bool MyAlgebra::Matrix::operator==(const Matrix& rhs) const
{
	if (row != rhs.row || coll != rhs.coll)
	{
		return false;
	}
	for (unsigned int i = 0; i < fullSize; ++i)
	{
		if (abs(data[i] - rhs.data[i]) > ALG_PRECISION)
		{
			return false;
		}
	}
	return true;
}

void MyAlgebra::Matrix::CreateData(bool randomInit)
{
	data = new float[(size_t)row * coll];
	if (randomInit)
	{
		for (unsigned int i = 0; i < row * coll; ++i)
		{
			data[i] = GENERATOR.Generate();
		}
	}
	else
	{
		const unsigned int canBeUnfolded = (fullSize / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;
		for (unsigned int i = 0; i < canBeUnfolded; i += VECTORIZATION_SIZE)
		{
			__m256 memDefVal = _mm256_set1_ps(DEFAULT_VALUE);
			_mm256_store_ps(data + i, memDefVal);
		}
		for (unsigned int i = canBeUnfolded; i < fullSize; ++i)
		{
			data[i] = DEFAULT_VALUE;
		}
	}
}

void MyAlgebra::Matrix::CopyData(const Matrix& rhs)
{
	if (row != rhs.row || coll != rhs.coll)
	{
		ClearData();
		row = rhs.row;
		coll = rhs.coll;
		fullSize = rhs.fullSize;
		CreateData(false);
	}
	const unsigned int canBeLoaded = (fullSize / VECTORIZATION_SIZE) * VECTORIZATION_SIZE;

	const register float* rhsDataPointer = rhs.data;
	for (unsigned int i = 0; i < canBeLoaded; i += VECTORIZATION_SIZE)
	{
		__m256 memRhsData = _mm256_load_ps(rhsDataPointer + i);
		_mm256_store_ps(data + i, memRhsData);
	}
	for (unsigned int j = canBeLoaded; j < fullSize; ++j)
	{
		data[j] = rhs.data[j];
	}
}

void MyAlgebra::Matrix::MoveData(Matrix& rhs)
{
	ClearData();
	data = rhs.data;
	row = rhs.row;
	coll = rhs.coll;
	fullSize = rhs.fullSize;
	rhs.data = nullptr;
	rhs.row = DEFAULT_VALUE;
	rhs.coll = DEFAULT_VALUE;
	rhs.fullSize = DEFAULT_VALUE;
}

void MyAlgebra::Matrix::ClearData()
{
	if (data != nullptr)
	{
		delete[] data;
		data = nullptr;
	}
}

MyAlgebra::Matrix MyAlgebra::operator*(float multiplier, const MyAlgebra::Matrix& rhs)
{
	return rhs * multiplier;
}