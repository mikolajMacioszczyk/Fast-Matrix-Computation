#include "MatrixException.h"

MyAlgebra::MatrixException::MatrixException(const std::string& operation, const std::string& cause)
	: operation(operation), cause(cause)
{
}

std::string MyAlgebra::MatrixException::GetOperation()
{
	return operation;
}

std::string MyAlgebra::MatrixException::GetCause()
{
	return cause;
}

std::string MyAlgebra::MatrixException::GetFullString()
{
	return "Excepion in operation: " + operation + ", caused by: " + cause;
}
