#pragma once
#include <string>

namespace MyAlgebra
{
	class MatrixException
	{
	public:
		MatrixException(const std::string& operation, const std::string& cause);

		std::string GetOperation();
		std::string GetCause();
		std::string GetFullString();

	private:
		std::string operation;
		std::string cause;
	};
}