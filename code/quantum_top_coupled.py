import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='4' # set number of MKL threads to run in parallel

#quspin_path = os.path.join(os.path.expanduser('~'),"quspin/basis_update/")
quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import obs_vs_time
from quspin.tools.evolution import evolve
from quspin.tools.Floquet import Floquet_t_vec

#from quspin.operators._oputils import matvec as _matvec
from quspin.operators._oputils import _get_matvec_function
from cpp_funcs import c_cross, c_dot

from numba import njit

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=22)
plt.tick_params(labelsize=18)

np.set_printoptions(suppress=True,precision=4)
np.random.seed(0)

##### classical kicked top

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

#H = 1.0/(2*J+1) * ( alpha*S_x + tau/(2*J+1)*S_z_dyn )
H = alpha*S_x + tau*S_z_dyn 


S_x_mat=S_x.tocsr()
S_y_mat=S_y.tocsr()
S_z_mat=S_z.tocsr()
_static_matvec=_get_matvec_function(S_y_mat)
	

def expectation(psi_1,psi_2,a,out=None,V_out=None):
	
	psi_1_conj=psi_1.conj()

	_static_matvec(S_x_mat, psi_2, out=V_out, a=+1.0, overwrite_out=True)
	#out[0]=psi_1_conj.dot(V_out)
	out[0]=c_dot(psi_1_conj,V_out).real

	_static_matvec(S_y_mat, psi_2, out=V_out, a=+1.0, overwrite_out=True)
	#out[1]=psi_1_conj.dot(V_out)
	out[1]=c_dot(psi_1_conj,V_out).real

	_static_matvec(S_z_mat, psi_2, out=V_out, a=+1.0, overwrite_out=True)
	#out[2]=psi_1_conj.dot(V_out)
	out[2]=c_dot(psi_1_conj,V_out).real

	out*=a

#	return out
	
@njit
def cross(v_1,v_2,out=None):
	out[0]+=v_1[1]*v_2[2] - v_1[2]*v_2[1]
	out[1]+=v_1[2]*v_2[0] - v_1[0]*v_2[2]
	out[2]+=v_1[0]*v_2[1] - v_1[1]*v_2[0]

#	return out

def sigma_dot_S(V,V_out,S,a):

	_static_matvec(S_x_mat, V, out=V_out, a=a*S[0], overwrite_out=False)
	_static_matvec(S_y_mat, V, out=V_out, a=a*S[1], overwrite_out=False)
	_static_matvec(S_z_mat, V, out=V_out, a=a*S[2], overwrite_out=False)

#Ns_plus_3=basis.Ns+3
#twice_Ns=2*basis.Ns
#twice_Ns_plus_3=2*basis.Ns+3

def EOM(t,V,Ns,gamma):

	V_dot=np.zeros_like(V)
	
	# auxiliary variables
	V_out=np.zeros(Ns,dtype=np.complex128)
	sigma_expt=np.zeros(3,dtype=np.float64)
	sigma_expt_lin=np.zeros(3,dtype=np.float64)

	expectation(V[:Ns],V[:Ns]        ,     gamma,out=sigma_expt    ,V_out=V_out)
	expectation(V[Ns+3:2*Ns+3],V[:Ns], 2.0*gamma,out=sigma_expt_lin,V_out=V_out)#.real

	
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
Uy=sp.linalg.expm(-1j*hbar_inv*theta*S_y_mat)
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
lyap_exp=np.zeros((time.N//m+1,),dtype=np.float64)
measure_times=np.zeros_like(lyap_exp)
k=0
for j,V in enumerate(V_t):

	S_lin=V[basis.Ns+3:2*basis.Ns+3]
	S=V[:basis.Ns]

	#lin=V[basis.Ns+3:2*basis.Ns+3] # V[2*basis.Ns+3:] #V[basis.Ns+3:] #
	#lin=2*expectation(V[basis.Ns+3:2*basis.Ns+3],V[:basis.Ns]).real
	#lin=np.outer(V[basis.Ns+3:2*basis.Ns+3], V[:basis.Ns].conj())
	#lin += lin.conj().T
	
	#print( expectation(S,S).real, V[basis.Ns:basis.Ns+3].real, 2.0*expectation(S_lin,S).real, V[2*basis.Ns+3:].real )
	#if j==5:
	#	exit()

	if j==0:
		lyap_exp[k]=0.0
		k+=1
		measure_times[0]=0.0
		last_time=0.0
	elif j%skip_steps==0: 
		n=np.linalg.norm(S_lin)
		#norm*=n
		#lyap_exp[k] = np.log(norm) /(1.0*time[j])
		lyap_exp[k] = (lyap_exp[k-1]*measure_times[k-1] + np.log(n) )/time[j]
		measure_times[k]=time[j]
		# normalize state: affects the ODE solver state
		S_lin/=n
		k+=1
		last_time=time[j]

		print(j, time.len, lyap_exp[k-1])

exit()

plt.plot(measure_times/time.T,lyap_exp)
plt.grid()
plt.xlabel('$\\ell$',)
plt.ylabel('$\\lambda_\\mathrm{max}(\\ell T)$')

title_str="$\\gamma={0:0.4f},\\ \\alpha={1:0.2f},\\ \\tau={2:0.2f},\\ \\omega={3:0.2f}$".format(gamma,alpha,tau,Omega)
plt.title(title_str,size=16)


plt.tight_layout()

#fig_name="lyapexp_vs_time-quantum_coupled-gamma={0:0.4f}_alpha={1:0.2f}_tau={2:0.2f}_omega={3:0.2f}_omega={4:d}.pdf".format(gamma,alpha,tau,Omega,N_T)
#plt.savefig(fig_name)

plt.show()