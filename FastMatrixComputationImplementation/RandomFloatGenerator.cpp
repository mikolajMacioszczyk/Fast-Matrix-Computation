#include "RandomFloatGenerator.h"
#include <cstdlib>
#include <ctime>

MyAlgebra::RandomFloatGenerator::RandomFloatGenerator()
{
	srand(time(NULL));
}

float MyAlgebra::RandomFloatGenerator::Generate()
{
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 4) - 2.0f;
}
