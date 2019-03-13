import sys,os
quspin_path = os.path.join(os.path.expanduser('~'),"quspin/basis_update/")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import obs_vs_time
from quspin.tools.evolution import evolve
from quspin.tools.Floquet import Floquet_t_vec

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True,precision=4)


##### classical kicked top

# H(t) = \alpha S^x + \tau (S^z)^2 * (periodic drive)

tau=6.0
alpha=1.5

gamma=0.01

Omega=2*np.pi
T=2*np.pi/Omega

def drive(t,Omega):
	return np.cos(Omega*t)
drive_args=[Omega]


S_str="1"
j=1.0
basis=spin_basis_1d(L=1,S=S_str)

#static=[ ['+', [[0.5*alpha,0]] ], ['-', [[0.5*alpha,0]] ] ]
dynamic=[['zz',[[1.0,0,0]],drive,drive_args] ]

no_checks=dict(check_herm=False,check_symm=False,check_pcon=False)

S_z_dyn=hamiltonian([],dynamic,basis=basis,dtype=np.complex128, **no_checks)

S_p=hamiltonian([['+',[[1.0, 0]] ]], [], basis=basis,dtype=np.complex128, **no_checks )
S_m=hamiltonian([['-',[[1.0, 0]] ]], [], basis=basis,dtype=np.complex128, **no_checks )
S_x= 0.5 *(S_p + S_m)
S_y=-0.5j*(S_p - S_m)
S_z=hamiltonian([['z',[[1.0, 0]] ]], [], basis=basis,dtype=np.complex128, **no_checks )

H = alpha*S_x + tau/(2*j+1)*S_z_dyn


S_x_mat=S_x.toarray()
S_y_mat=S_y.toarray()
S_z_mat=S_z.toarray()


def expectation(psi):
	return np.array( [S_x.expt_value(psi), S_y.expt_value(psi), S_z.expt_value(psi)] )

def ext_field_op(S):
	return S[0]*S_x_mat + S[1]*S_y_mat + S[2]*S_z_mat


def EOM(t,V,Ns,gamma):

	V_dot=np.zeros_like(V)
	
	V_dot[:Ns] = -1j*gamma*ext_field_op(V[Ns:]).dot(V[:Ns]) + H._hamiltonian__SO(t,V[:Ns])
	V_dot[Ns:] = gamma*np.cross( expectation(V[:Ns]), V[Ns:] )

	return V_dot

EOM_cont_args=[basis.Ns,gamma]


S_q=np.array([0.0,0.0,1.0])
S_cl=np.array([0.0,0.0,1.0])
S_0=np.concatenate([S_q,S_cl]).astype(np.complex128)

S_q=np.array([0.01,0.0,1.0])
S_q/=np.linalg.norm(S_q)
S_cl=np.array([0.0,0.0,1.0])
S_1=np.concatenate([S_q,S_cl]).astype(np.complex128)


time=Floquet_t_vec(Omega,100)  #np.linspace(0.0,20.0*T,101)
S_t=evolve(S_0,time[0],time,EOM,f_params=EOM_cont_args,iterate=False,atol=1E-12,rtol=1E-12)
S_t2=evolve(S_1,time[0],time,EOM,f_params=EOM_cont_args,iterate=False,atol=1E-12,rtol=1E-12)


S_q_t = S_t[:basis.Ns]
S_cl_t = S_t[basis.Ns:]

S_q_t2 = S_t2[:basis.Ns]
S_cl_t2 = S_t2[basis.Ns:]

plt.plot(time.vals, np.linalg.norm(S_q_t,axis=0) )
plt.plot(time.vals, np.linalg.norm(S_cl_t,axis=0) )
plt.show()

plt.plot(time.vals, np.abs(S_q_t[0] - S_q_t2[0] ) )
plt.plot(time.vals, np.abs(S_q_t[1] - S_q_t2[1] ) )
plt.plot(time.vals, np.abs(S_q_t[2] - S_q_t2[2] ) )
plt.show()

plt.plot(time.vals, np.abs(S_cl_t[0] - S_cl_t2[0] ) )
plt.plot(time.vals, np.abs(S_cl_t[1] - S_cl_t2[1] ) )
plt.plot(time.vals, np.abs(S_cl_t[2] - S_cl_t2[2] ) )

plt.show()











