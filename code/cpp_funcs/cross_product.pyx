# cython: language_level=2
## cython: profile=True
# distutils: language=c++

cimport cython
#import numpy as _np
cimport numpy as np


cdef extern from "complex_ops.h":
	cdef cppclass npy_cdouble_wrapper:
		pass

cdef extern from "cross.h":
	void c_cross_product(const double *,const npy_cdouble_wrapper *,npy_cdouble_wrapper *)
	np.complex128_t c_dot_product(np.complex128_t *,np.complex128_t *, int)
	

def c_cross(np.ndarray v1, np.ndarray v2, np.ndarray out):
	# create void pinters
	# cast pointers with angle brackets to <npy> to datatype
	
	cdef void * v1_ptr=np.PyArray_DATA(v1)
	cdef void * v2_ptr=np.PyArray_DATA(v2)
	cdef void * out_ptr=np.PyArray_DATA(out)

	c_cross_product(< const double *> v1_ptr, < const npy_cdouble_wrapper *> v2_ptr, < npy_cdouble_wrapper *> out_ptr)


def c_dot(np.ndarray v1, np.ndarray v2, int N):
	#cdef int N=v2.shape[0]
	
	cdef void * v1_ptr=np.PyArray_DATA(v1)
	cdef void * v2_ptr=np.PyArray_DATA(v2)
	
	return c_dot_product(< const np.complex128_t *> v1_ptr, < const np.complex128_t *> v2_ptr, N)


