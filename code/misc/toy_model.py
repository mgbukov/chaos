import sys,os
quspin_path = os.path.join(os.path.expanduser('~'),"quspin/basis_update/")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import obs_vs_time
from quspin.tools.evolution import evolve
from quspin.tools.Floquet import Floquet_t_vec

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True,precision=3)


##### classical kicked top

# H(t) = \alpha S^x + \tau (S^z)^2 * (periodic kicks)

tau=6.0
alpha=1.5
Omega=2*np.pi
T=2*np.pi/Omega

cosa=np.cos(alpha)
sina=np.sin(alpha)


def EOM(S,tau,cosa,sina):

	Sx=S[0]
	diff_1=S[1]*cosa - S[2]*sina
	diff_2=S[1]*sina + S[2]*cosa
	phase=tau*diff_2
	

	S[0]=Sx*np.cos(phase) - diff_1*np.sin(phase)	
	S[1]=Sx*np.sin(phase) + diff_1*np.cos(phase)
	S[2]=diff_2

	return S

EOM_args=[tau,cosa,sina]


def top_evolve(S_0,N_T=30):

	S_t=np.zeros((3,N_T+1),dtype=np.float64)
	S_t[:,0]=S_0

	S=S_0.copy()
	for j in range(N_T):
		S=EOM(S,*EOM_args)
		S_t[:,j+1]=S

	return S_t, np.arange(N_T+1)



def EOM_cont(t,S,tau,alpha,Omega):

	S_new=S.copy()

	S_new[0] = -2.0*tau*S[1]*S[2]*np.cos(Omega*t)
	S_new[1] = -alpha*S[2] + 2.0*tau*S[0]*S[2]*np.cos(Omega*t)
	S_new[2] =  alpha*S[1]

	return S_new

EOM_cont_args=[tau,alpha,Omega]

S_0_1=np.array([0.23,0.0,0.69]) 
S_0_1/=np.linalg.norm(S_0_1)

S_0_2=np.array([0.23,1E-7,0.69]) 
S_0_2/=np.linalg.norm(S_0_2)

dS_0 = S_0_1 - S_0_2

#S_t_1, time=top_evolve(S_0_1)
#S_t_2, time=top_evolve(S_0_2)


time=Floquet_t_vec(Omega,30)  #np.linspace(0.0,20.0*T,101)
S_t_1=evolve(S_0_1,time[0],time,EOM_cont,f_params=EOM_cont_args,real=True)
S_t_2=evolve(S_0_2,time[0],time,EOM_cont,f_params=EOM_cont_args,real=True)



angle = np.arccos( np.einsum('ij,ij->j',S_t_1,S_t_2) )

#'''
plt.plot( time/time.T, S_t_1[0,:] )
plt.plot( time/time.T, S_t_2[0,:] )
plt.plot( time/time.T, np.linalg.norm(S_t_2,axis=0) )
plt.show()
#'''

plt.plot(time.strobo.vals,angle[time.strobo.inds]/np.linalg.norm(dS_0),'*')
plt.plot(time.strobo.vals,np.exp(0.5*time.strobo.vals),'-')
plt.yscale('log')
plt.show()




exit()




##### qubit system

basis=spin_basis_1d(L=1,pauli=True)

gamma=1.0 # soupling strength
sc_list=[[gamma,0]]

Omega=0.001
T=2*np.pi/Omega
def ext_field_x(t,Omega):
	return 1.0/np.sqrt(2.0)*np.cos(Omega*t)

def ext_field_y(t,Omega):
	return 1.0/np.sqrt(2.0)*np.sin(Omega*t)

ext_field_args=[Omega]

static=[] #[['z',sc_list]]
dynamic=[['x',sc_list,ext_field_x,ext_field_args],['y',sc_list,ext_field_y,ext_field_args]]

no_checks=dict(check_herm=False,check_symm=False,check_pcon=False)

H=hamiltonian(static,dynamic,basis=basis,dtype=np.complex128, **no_checks)

sigma_x=hamiltonian([['x',sc_list]] ,[], basis=basis,dtype=np.complex128, **no_checks )
sigma_y=hamiltonian([['y',sc_list]] ,[], basis=basis,dtype=np.complex128, **no_checks )
sigma_z=hamiltonian([['z',sc_list]] ,[], basis=basis,dtype=np.complex128, **no_checks )

E,V=H.eigh(time=0)

psi_0=V[:,0]

N_T=11
time=np.linspace(0.0,T,N_T)
psi_t=H.evolve(psi_0,0.0,time,iterate=True)

# instanneous state
psi_inst=np.zeros((2,N_T),dtype=np.complex128)

for j,t in enumerate(time):
	E,V=H.eigh(time=t)
	psi_inst[:,j]=V[:,0]


Obs_dict=dict(sigma_x=sigma_x,sigma_y=sigma_y,sigma_z=sigma_z)

obs_exact=obs_vs_time(psi_t,time,Obs_dict,return_state=True)
obs_inst=obs_vs_time(psi_inst,time,Obs_dict,return_state=True)

#print(obs_exact['sigma_x'])
#print(obs_inst['sigma_x'])






