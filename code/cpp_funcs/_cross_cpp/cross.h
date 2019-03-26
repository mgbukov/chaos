//#ifndef _CROSS_C_H
//#define _CROSS_C_H
#include "openmp.h"
#include "complex_ops.h"
#include <complex>
using namespace std;



void c_cross_product(const double v_1[],
					 const npy_cdouble_wrapper v_2[],
						   npy_cdouble_wrapper out[]
	)
{
	out[0]+=v_1[1]*v_2[2] - v_1[2]*v_2[1];
	out[1]+=v_1[2]*v_2[0] - v_1[0]*v_2[2];
	out[2]+=v_1[0]*v_2[1] - v_1[1]*v_2[0];
}




std::complex<double> c_dot_product(const std::complex<double> v_1[],
							     const std::complex<double> v_2[],
							     const int N
	)
{	
	std::complex<double> result=0;
	#pragma omp parallel
	{
		#pragma omp for schedule(static) // constant load operation
		for(int j=0;j<N;j++){
			result+=std::conj(v_1[j])*v_2[j];
		}
	}
	

	return result;
}