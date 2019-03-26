import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='4' # set number of MKL threads to run in parallel

quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import evolve
from quspin.tools.Floquet import Floquet_t_vec

# quspin matvec
from quspin.operators._oputils import _get_matvec_function
# custom cpp functions
from cpp_funcs import c_cross, c_dot

import numpy as np
import scipy as sp

np.set_printoptions(suppress=True,precision=4)
np.random.seed(0)

##### kicked top

# H(t) = \alpha S^x + \tau (S^z)^2 * (periodic drive)

J=1 # spin value
hbar=1.0/J
hbar_inv=1.0/hbar
gamma=1.00
tau=0.0 #7.0
alpha=1.5
Omega=2*np.pi
T=2*np.pi/Omega


def drive(t,Omega):
	return np.cos(Omega*t)
drive_args=[Omega]

S_str="{:d}".format(J)
basis=spin_basis_1d(L=1,S=S_str)

dynamic=[['zz',[[hbar**2,0,0]],drive,drive_args] ]

no_checks=dict(check_herm=False,check_symm=False,check_pcon=False)

S_z_dyn=hamiltonian([],dynamic,basis=basis,dtype=np.complex128, **no_checks)

S_p=hamiltonian([['+',[[hbar, 0]] ]], [], basis=basis,dtype=np.complex128, **no_checks )
S_m=hamiltonian([['-',[[hbar, 0]] ]], [], basis=basis,dtype=np.complex128, **no_checks )
S_x= 0.5 *(S_p + S_m)
S_y=-0.5j*(S_p - S_m)
S_z=hamiltonian([['z',[[hbar, 0]] ]], [], basis=basis,dtype=np.complex128, **no_checks )

H = alpha*S_x + tau*S_z_dyn 


S_x_mat=S_x.tocsr()
S_y_mat=S_y.tocsr()
S_z_mat=S_z.tocsr()
_static_matvec=_get_matvec_function(S_y_mat)
	

def expectation(psi_1,psi_2,a,Ns,out=None,V_tmp=None):
	'''
	<psi_1|\\vec S\\psi_2>, i.e. psi_1 will be conjugated inside the function c_dot
	'''
	
	_static_matvec(S_x_mat, psi_2, out=V_tmp, a=+1.0, overwrite_out=True)
	out[0]=c_dot(psi_1,V_tmp,Ns).real

	_static_matvec(S_y_mat, psi_2, out=V_tmp, a=+1.0, overwrite_out=True)
	out[1]=c_dot(psi_1,V_tmp,Ns).real

	_static_matvec(S_z_mat, psi_2, out=V_tmp, a=+1.0, overwrite_out=True)
	out[2]=c_dot(psi_1,V_tmp,Ns).real

	out*=a
	

def sigma_dot_S(V,V_out,S,a):

	_static_matvec(S_x_mat, V, out=V_out, a=a*S[0], overwrite_out=False)
	_static_matvec(S_y_mat, V, out=V_out, a=a*S[1], overwrite_out=False)
	_static_matvec(S_z_mat, V, out=V_out, a=a*S[2], overwrite_out=False)

#Ns_plus_3=basis.Ns+3
#twice_Ns=2*basis.Ns
#twice_Ns_plus_3=2*basis.Ns+3

def EOM(t,V,Ns,gamma):
	'''
	V[:Ns] -- \\psi
	V[Ns:Ns+3] -- S
	V[Ns+3:2*Ns+3] -- \\delta\\psi 
	V[2*Ns+3:] -- \\delta S
	'''

	V_dot=np.zeros_like(V)
	
	# auxiliary variables
	V_tmp=np.zeros(Ns,dtype=np.complex128)
	sigma_expt=np.zeros(3,dtype=np.float64)
	sigma_expt_lin=np.zeros(3,dtype=np.float64)

	expectation(V[:Ns],V[:Ns]        ,     gamma,Ns,out=sigma_expt    ,V_tmp=V_tmp)
	expectation(V[Ns+3:2*Ns+3],V[:Ns], 2.0*gamma,Ns,out=sigma_expt_lin,V_tmp=V_tmp)
	
	# evolve quantum state vector
	V_dot[:Ns] = hbar_inv*H._hamiltonian__SO(t,V[:Ns],V_dot[:Ns]) # -1j*H(t)*|psi>
	
	# coupling term
	#V_dot[:Ns]     -= 1j/hbar*sigma_dot.dot(V[:Ns])
	sigma_dot_S(V[:Ns],V_dot[:Ns],V[Ns:Ns+3],-1j*gamma*hbar_inv)
	#cross(sigma_expt,V[Ns:Ns+3],out=V_dot[Ns:Ns+3])
	c_cross(sigma_expt,V[Ns:Ns+3],out=V_dot[Ns:Ns+3])
	
	# evolve linearization
	V_dot[Ns+3:2*Ns+3]  = hbar_inv*H._hamiltonian__SO(t,V[Ns+3:2*Ns+3],V_dot[Ns+3:2*Ns+3]) # -1j*H(t)*|psi_lin>
	
	# coupling term
	#V_dot[Ns+3:2*Ns+3] -= 1j/hbar*sigma_dot.dot(V[Ns+3:2*Ns+3]) # S sigma |psi_lin>
	sigma_dot_S(V[Ns+3:2*Ns+3],V_dot[Ns+3:2*Ns+3],V[Ns:Ns+3],-1j*gamma*hbar_inv) # S sigma |psi_lin>
	#V_dot[Ns+3:2*Ns+3] -= 1j/hbar*sigma_lin_dot.dot(V[:Ns]) # S_lin sigma |psi>
	sigma_dot_S(V[:Ns],V_dot[Ns+3:2*Ns+3],V[2*Ns+3:],-1j*gamma*hbar_inv) # S_lin sigma |psi>
	
	#cross(sigma_expt,V[2*Ns+3:],out=V_dot[2*Ns+3:]) # <psi|sigma|psi> x S_lin
	#cross(sigma_expt_lin,V[Ns:Ns+3],out=V_dot[2*Ns+3:]) # 2 Re <\delta psi|sigma|psi_lin> x S

	c_cross(sigma_expt,V[2*Ns+3:]    ,out=V_dot[2*Ns+3:]) # <psi|sigma|psi> x S_lin
	c_cross(sigma_expt_lin,V[Ns:Ns+3],out=V_dot[2*Ns+3:]) # 2 Re <\delta psi|sigma|psi_lin> x S

	return V_dot

EOM_cont_args=[basis.Ns,gamma]


# initial concatenatendition
psi_0=np.zeros(basis.Ns)
psi_0[0]=1.0
# rotate initial state
theta=np.arctan(0.23/0.69)
Uy=sp.linalg.expm(-1j*hbar_inv*theta*S_y_mat.tocsc())
psi_0=Uy.dot(psi_0)

# classical initial state
S_cl_0=np.array([0,0,1]) # (x,y,z)-components

S_0=np.concatenate([psi_0,S_cl_0])

# initial conditions for auxiliary EOM
psi_lin_0 = np.random.normal(size=basis.Ns).astype(np.complex128)
psi_lin_0/=np.linalg.norm(psi_lin_0)
psi_lin_0-=psi_lin_0.conj().dot(psi_0)*psi_0


S_aux_0=np.zeros(basis.Ns+3,dtype=np.complex128)
S_aux_0[:basis.Ns] = psi_lin_0
S_aux_0[basis.Ns:] = np.ones(3)/np.sqrt(3)

# total initial state
V_0=np.concatenate([S_0,S_aux_0])


N_T=200 #10 #200
time=Floquet_t_vec(Omega,N_T)  #np.linspace(0.0,20.0*T,101)
V_t=evolve(V_0,time[0],time,EOM,f_params=EOM_cont_args,iterate=True,atol=1E-12,rtol=1E-12)


# calculate linear evolution
m=1
skip_steps=m*time.len_T

lyap_exp_dpsi=np.zeros((time.N//m+1,),dtype=np.float64)
lyap_exp_dS=np.zeros_like(lyap_exp_dpsi)
lyap_exp_dx=np.zeros_like(lyap_exp_dpsi)
lyap_exp_dO=np.zeros_like(lyap_exp_dpsi)

dO=np.zeros(3,dtype=np.float64)
V_tmp=np.zeros(basis.Ns,dtype=np.complex128)

measure_times=np.zeros_like(lyap_exp_dpsi)
k=0
for j,V in enumerate(V_t):

	dpsi=V[basis.Ns+3:2*basis.Ns+3]
	dS=V[2*basis.Ns+3:]
	dx=V[basis.Ns+3:]
	expectation(dpsi,dpsi,1.0,basis.Ns,out=dO,V_tmp=V_tmp)
	
	if j==0:
		lyap_exp_dpsi[k]=0.0
		lyap_exp_dS[k]=0.0
		lyap_exp_dx[k]=0.0
		lyap_exp_dO[k]=0.0

		k+=1
		measure_times[0]=0.0
		last_time=0.0

	elif j%skip_steps==0:

		n_dx   = np.linalg.norm(dx) 
		n_dpsi = np.linalg.norm(dpsi)
		n_dS   = np.linalg.norm(dS)
		n_dO   = np.linalg.norm(dO)

		
		lyap_exp_dx[k]   = (lyap_exp_dx[k-1]   * measure_times[k-1] + np.log(n_dx)   				  )/time[j]
		lyap_exp_dpsi[k] = (lyap_exp_dpsi[k-1] * measure_times[k-1] + np.log(n_dpsi) )/time[j]
		lyap_exp_dS[k]   = (lyap_exp_dS[k-1]   * measure_times[k-1] + np.log(n_dS)     )/time[j]
		lyap_exp_dO[k]   = (lyap_exp_dO[k-1]   * measure_times[k-1] + np.log(n_dO)     )/time[j]
		
		# append time
		measure_times[k]=time[j]

		# normalize state: affects the ODE solver state
		#dx/=n_dx
		dpsi/=n_dpsi
		#dS/=n_dS
		#dO/=n_dO

	
		k+=1
		last_time=time[j]

		print(j, time.len, lyap_exp_dpsi[k-1], lyap_exp_dS[k-1], lyap_exp_dx[k-1], lyap_exp_dO[k-1])



title_str="$\\gamma={0:0.4f},\\ \\alpha={1:0.2f},\\ \\tau={2:0.2f},\\ \\omega={3:0.2f}$".format(gamma,alpha,tau,Omega)
