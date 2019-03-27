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
from six import iteritems
import pickle

np.set_printoptions(suppress=True,precision=4)
np.random.seed(0)


###### model parameters

# kicked top
# H(t) = \alpha S^x + \tau (S^z)^2 * (periodic drive)

### coupling constants
J=1 # spin value, e.g. spin-1, spin-3/2 or spin-J
hbar=1.0/J
hbar_inv=1.0/hbar
gamma=1.00
tau=7.0 #7.0
alpha=1.5
Omega=2*np.pi 

params_str="gamma={0:0.6f}_alpha={1:0.2f}_tau={2:0.2f}_omega={3:0.2f}".format(gamma,alpha,tau,Omega)

### time vector
N_T=20 # number of driving cycles to evolve for
ti=0.0 # initial value
time=Floquet_t_vec(Omega,N_T) # choose time points to hit strobo periodsl N_T cycles in total
time._vals+=ti

time_str="ti={0:0.2f}_tf={1:0.2f}".format(time[0],time[-1])


######
save_data=True
use_checkpoints=True

# file name
model_str="q_top"
ext_str='.pkl'
def create_file_name(model_str,time_str,params_str,ext_str):
	return model_str+"-"+time_str+"-"+params_str+ext_str
file_name=create_file_name(model_str,time_str,params_str,ext_str)

# read in local directory path
str1=os.getcwd()
str2=str1.split('\\')
n=len(str2)
_dir = str2[n-1]

# check if data directory exists and create it
data_dir = _dir+"/data/"
if not os.path.exists(data_dir):
	os.makedirs(data_dir)

# check if simulation results exist
def check_for_simulation_data():
	# check if same output file exists
	if os.path.exists(data_dir+file_name):
		print('data file with simulation data already exists')
		print('exiting...\n')
		exit()

check_for_simulation_data()

###### define quantum Hamiltonian and operators

def drive(t,Omega):
	return np.cos(Omega*t)
drive_args=[Omega]

S_str="{:d}".format(J)
basis=spin_basis_1d(L=1,S=S_str)
# dynamic list for (S^z)^2 * (periodic drive)
dynamic=[['zz',[[hbar**2,0,0]],drive,drive_args] ]

no_checks=dict(check_herm=False,check_symm=False,check_pcon=False)

# construct Hamiltonian and spin operators
S_z_dyn=hamiltonian([],dynamic,basis=basis,dtype=np.complex128, **no_checks)

S_p=hamiltonian([['+',[[hbar, 0]] ]], [], basis=basis,dtype=np.complex128, **no_checks )
S_m=hamiltonian([['-',[[hbar, 0]] ]], [], basis=basis,dtype=np.complex128, **no_checks )
S_x= 0.5 *(S_p + S_m)
S_y=-0.5j*(S_p - S_m)
S_z=hamiltonian([['z',[[hbar, 0]] ]], [], basis=basis,dtype=np.complex128, **no_checks )

H = alpha*S_x + tau*S_z_dyn 

# get operators as sparce matrices
S_x_mat=S_x.tocsr()
S_y_mat=S_y.tocsr()
S_z_mat=S_z.tocsr()



###### determine appropriate matvec functions

_static_matvec=_get_matvec_function(S_y_mat)
_dynamic_matvec = {}
for func,Hd in iteritems(H._dynamic):
	_dynamic_matvec[func] = _get_matvec_function(Hd)
	


###### auxiliary functions for EOM

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


def schroedinger_evolution(time,V,V_out,a=1.0):
	"""
	args:
		H, quspin hamiltonian object
		V, the vector to multiple with
		V_out, the vector to use with output.
		time, the time to evalute drive at.
	"""

	# apply static Hamiltonian
	_static_matvec(H._static,V,out=V_out,overwrite_out=True)
	# apply dynamic Hamiltonian
	for func,Hd in iteritems(H._dynamic):
		_dynamic_matvec[func](Hd,V,a=func(time),out=V_out,overwrite_out=False)
	# apply constants
	V_out *= -1j*a


###### define EOM

def EOM(t,V,Ns,gamma):
	'''
	Defines joint EOM: nonlinear EOM (for x) and chaotic EOM (for dx). 

	V[:Ns] -- \\psi
	V[Ns:Ns+3] -- S
	V[Ns+3:2*Ns+3] -- \\delta\\psi 
	V[2*Ns+3:] -- \\delta S

	Ns -- # of states in Hilbert space
	gamma -- quantum/classical coupling
	'''

	V_dot=np.zeros_like(V)
	
	##### preallocate auxiliary variables which appear in both the x and dx EOM
	V_tmp=np.zeros(Ns,dtype=np.complex128)
	sigma_expt=np.zeros(3,dtype=np.float64)
	sigma_expt_lin=np.zeros(3,dtype=np.float64)
	expectation(V[:Ns],V[:Ns]        ,     gamma,Ns,out=sigma_expt    ,V_tmp=V_tmp)
	expectation(V[Ns+3:2*Ns+3],V[:Ns], 2.0*gamma,Ns,out=sigma_expt_lin,V_tmp=V_tmp)
	
	##### evolve nonlinear EOM for x
	#
	# EOM for \\psi (coupling to S in the following line)
	schroedinger_evolution(t,V[:Ns],V_dot[:Ns],a=hbar) # -1j/hbar*H(t)*|psi>
	# coupling term
	sigma_dot_S(V[:Ns],V_dot[:Ns],V[Ns:Ns+3],-1j*gamma*hbar_inv) #V_dot[:Ns]     -= 1j/hbar*sigma_dot.dot(V[:Ns])
	#
	# EOM for S
	c_cross(sigma_expt,V[Ns:Ns+3],out=V_dot[Ns:Ns+3]) # gamma*<\\sigma> x S
	
	##### evolve linearized EOM for dx
	#
	# EOM for \\delta\\psi (coupling in the followign line)
	schroedinger_evolution(t,V[Ns+3:2*Ns+3],V_dot[Ns+3:2*Ns+3],a=hbar_inv) # -1j/hbar*H(t)|psi_lin>
	# coupling term
	#V_dot[Ns+3:2*Ns+3] -= 1j/hbar*sigma_dot.dot(V[Ns+3:2*Ns+3])
	sigma_dot_S(V[Ns+3:2*Ns+3],V_dot[Ns+3:2*Ns+3],V[Ns:Ns+3],-1j*gamma*hbar_inv) # S sigma |psi_lin>
	#V_dot[Ns+3:2*Ns+3] -= 1j/hbar*sigma_lin_dot.dot(V[:Ns])
	sigma_dot_S(V[:Ns],V_dot[Ns+3:2*Ns+3],V[2*Ns+3:],-1j*gamma*hbar_inv) # S_lin sigma |psi>
	#
	# EOM for \\delta S
	#cross(sigma_expt,V[2*Ns+3:],out=V_dot[2*Ns+3:]) # <psi|sigma|psi> x S_lin
	c_cross(sigma_expt,V[2*Ns+3:]    ,out=V_dot[2*Ns+3:]) # <psi|sigma|psi> x S_lin
	#cross(sigma_expt_lin,V[Ns:Ns+3],out=V_dot[2*Ns+3:]) # 2 Re <\delta psi|sigma|psi_lin> x S
	c_cross(sigma_expt_lin,V[Ns:Ns+3],out=V_dot[2*Ns+3:]) # 2 Re <\delta psi|sigma|psi_lin> x S

	return V_dot

EOM_args=[basis.Ns,gamma] # additional arguments for EOM



###### define initial conditions

### initial condition for nonlinear EOM
#
# initial condition \\psi(0): a spin-coherent state
psi_0=np.zeros(basis.Ns)
psi_0[0]=1.0
# rotate initial state
theta=np.arctan(0.23/0.69)
Uy=sp.linalg.expm(-1j*hbar_inv*theta*S_y_mat.tocsc())
psi_0=Uy.dot(psi_0)

# initial condition for classical S(0)
S_0=np.array([0.0,0.0,1.0]) # (x,y,z)-components

# total initial condition for nonlinear variable x=[\\psi,S]
x_0=np.concatenate([psi_0,S_0])

### initial condition for linearized EOM
#
# initial condition \\delta\\psi(0): needs to hit the direction of maximum lyapunov exponent
dpsi_0 = np.random.normal(size=basis.Ns).astype(np.complex128) # random state
dpsi_0/=np.linalg.norm(dpsi_0) # noprmalize state
dpsi_0-=dpsi_0.conj().dot(psi_0)*psi_0 # make orthogonal to \\psi

# initial condition for the linearized classical dS(0): needs to hit the direction of maximum lyapunov exponent
dS_0=np.ones(3)/np.sqrt(3.0)

dx_0=np.zeros(basis.Ns+3,dtype=np.complex128)
dx_0[:basis.Ns] = dpsi_0
dx_0[basis.Ns:] = dS_0

# total initial state (x,dx)
V_0=np.concatenate([x_0,dx_0])



###### compute Lyapunov exponents

# checks if checkpoint exists
using_checkpoint=False
all_files=os.listdir(data_dir)
for file in all_files:
	if ext_str in file and params_str in file and model_str in file:
		# remove all parts but the times
		file_time_str=file.replace(ext_str,'').replace(params_str,'').replace(model_str,'').replace('-','')
		# split initial and final times
		ti_str,tf_str=file_time_str.split('_')
		# etract initial and final time values
		ti_file_value=float(ti_str.split('=')[-1])
		tf_file_value=float(tf_str.split('=')[-1])
		
		if tf_file_value < time[-1]: #ti_file_value > time[0]
			print("\ncheckpoint found for these model parameters\n")
			if use_checkpoints:
				# load data
				with open(data_dir+file,'rb') as f:
					measure_times_file, lyap_exp_dx_file, V_file, V_0_file = pickle.load(f)

				using_checkpoint=True
				break;

			else:
				print("ignoring checkpoint data...\n")


# adjust times vector
if using_checkpoint:
	print("using checkpoint data...\n")
	N_T=int( (time[-1]-tf_file_value)/time.T )
	time=Floquet_t_vec(Omega,N_T)
	time._vals+=tf_file_value	

	# adust file name
	time_str="ti={0:0.2f}_tf={1:0.2f}".format(time[0],time[-1])
	file_name=create_file_name(model_str,time_str,params_str,ext_str)

check_for_simulation_data()

# calculate linear evolution
m=1
skip_steps=m*time.len_T # normalize linearized vector every skip_steps

lyap_exp_dx=np.zeros((time.N//m+1,),dtype=np.float64)
measure_times=np.zeros_like(lyap_exp_dx)

if using_checkpoint:
	V_0=V_file.copy()
	lyap_exp_dx[0]=lyap_exp_dx_file[-1]
	measure_times[0]=measure_times_file[-1]

# create generator for solution of EOM
V_t=evolve(V_0,time[0],time,EOM,f_params=EOM_args,iterate=True,atol=1E-12,rtol=1E-12)

k=1
for j,V in enumerate(V_t):

	# load linearized solution
	dx=V[basis.Ns+3:]

	if j%skip_steps==0 and j>0:
		# normalize state: affects the ODE solver state V thru the memory view
		n_dx   = np.linalg.norm(dx) 
		dx/=n_dx
		# compute Lyapunov exponent
		lyap_exp_dx[k]   = (lyap_exp_dx[k-1]   * measure_times[k-1] + np.log(n_dx)    )/time[j]
		
		# append time
		measure_times[k]=time[j]
		# update aux params	
		k+=1

		# print current exponent
		data_tuple=(j, time.len-1, lyap_exp_dx[k-1])
		print("finished {0:d}/{1:d} steps, lyap_exp={2:0.6f}".format(*data_tuple))

#exit()
if save_data:
	pickle.dump([measure_times, lyap_exp_dx, V, V_0], open(data_dir+file_name, "wb" ) )

