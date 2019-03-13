import sys,os
quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import obs_vs_time
from quspin.tools.evolution import evolve
from quspin.tools.Floquet import Floquet_t_vec

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=22)
plt.tick_params(labelsize=18)

np.set_printoptions(suppress=True,precision=4)


##### classical kicked top

# H(t) = \alpha S^x + \tau (S^z)^2 * (periodic kicks)

gamma=1.00
tau=0.0 #7.0
alpha=1.5
Omega=2*np.pi
T=2*np.pi/Omega


def EOM_max(t,S,tau,alpha,Omega,gamma):
	''' S = [S, M] contains both the spin vector and its linearization. '''

	S_new=np.zeros_like(S)

	'''
	# evolve spin vector
	S_new[0] = -2.0*tau*S[1]*S[2]*np.cos(Omega*t)
	S_new[1] = -alpha*S[2] + 2.0*tau*S[0]*S[2]*np.cos(Omega*t)
	S_new[2] =  alpha*S[1]

	# evolve linearization
	Jac_new = np.array([    [0.0, -2.0*tau*S[2]*np.cos(Omega*t), -2.0*tau*S[1]*np.cos(Omega*t)],
							[2.0*tau*S[2]*np.cos(Omega*t), 0.0, -alpha + 2.0*tau*S[0]*np.cos(Omega*t)],
							[0.0, alpha, 0.0]
						])

	S_new[6:9] = Jac_new.dot(S[6:9])

	'''

	# cross product
	S_cross_sigma = gamma*np.cross( S[0:3], S[3:6])
	S_cross_sigma_aux = gamma*( np.cross(S[3:6],S[6:9]) - np.cross(S[0:3],S[9:]) )
	#S_cross_sigma_aux2 = gamma*(  - np.cross(S[0:3],S[9:]) )
	#S_cross_sigma_aux3 = gamma*( np.cross(S[3:6],S[6:9]) )

	# evolve spin vector
	S_new[0] =             - 2.0*tau*S[1]*S[2]*np.cos(Omega*t)             
	S_new[1] = -alpha*S[2] + 2.0*tau*S[0]*S[2]*np.cos(Omega*t) 
	S_new[2] =  alpha*S[1]                                    
	# coupling term
	S_new[0:3] -= S_cross_sigma
	S_new[3:6]  = S_cross_sigma


	# evolve linearization
	Jac_new = np.array([    [0.0,                         -2.0*tau*S[2]*np.cos(Omega*t), -2.0*tau*S[1]*np.cos(Omega*t)        ],
							[2.0*tau*S[2]*np.cos(Omega*t), 0.0,                          -alpha + 2.0*tau*S[0]*np.cos(Omega*t)],
							[0.0,                          alpha,                         0.0                                 ]
						])


	S_new[6:9] = Jac_new.dot(S[6:9]) 
	S_new[6:9]+=  S_cross_sigma_aux
	S_new[9:]  = -S_cross_sigma_aux


	return S_new


EOM_cont_args=[tau,alpha,Omega,gamma]

# initial conditions for chaotic EOM
S_top_0=np.array([0.23,0.0,0.69])
S_top_0/=np.linalg.norm(S_top_0)

S_cl_0=np.array([0.0,0.0,1.0])

S_tot_0 = np.concatenate([S_top_0,S_cl_0])

# initial conditions for auxiliary EOM
S_aux_0 = np.array([0.1331, 0.0, -0.0444,    1.0/np.sqrt(3.0),1.0/np.sqrt(3.0),1.0/np.sqrt(3.0)])

S_0=np.concatenate([S_tot_0,S_aux_0])

N_T=200
time=Floquet_t_vec(Omega,N_T) 
SS_t=evolve(S_0,time[0],time,EOM_max,f_params=EOM_cont_args,real=True,iterate=True,atol=1E-14,rtol=1E-14)


# calculate linear evolution
m=1
skip_steps=m*time.len_T
lyap_exp=np.zeros((time.N//m+1,),dtype=np.float64)
measure_times=np.zeros_like(lyap_exp)
k=0
for j,SS in enumerate(SS_t):

	S_lin=SS[6:9]
	S=SS[0:3]

	lin=SS[6:9] #SS[9:] #SS[6:]
	
	#print(S,SS[3:6],S_lin,SS[9:])
	#if j==5:
	#	exit()

	if j==0:
		lyap_exp[k]=0.0
		k+=1
		measure_times[0]=0.0
		last_time=0.0
	elif j%skip_steps==0: 
		#n=np.linalg.norm(S_lin)
		n=np.linalg.norm(lin)		
		#norm*=n
		#lyap_exp[k] = np.log(norm) /(1.0*time[j])
		lyap_exp[k] = (lyap_exp[k-1]*measure_times[k-1] + np.log(n) )/time[j]
		measure_times[k]=time[j]
		# normalize state: affects the ODE solver state
		S_lin/=n
		k+=1
		last_time=time[j]

		print(j, time.len, lyap_exp[k-1])


print(SS[:3], SS[6:9])
print(SS[3:6], SS[9:])
		
plt.plot(measure_times/time.T,lyap_exp)
plt.grid()
plt.xlabel('$\\ell$',)
plt.ylabel('$\\lambda_\\mathrm{max}(\\ell T)$')

title_str="$\\gamma={0:0.4f},\\ \\alpha={1:0.2f},\\ \\tau={2:0.2f},\\ \\omega={3:0.2f}$".format(gamma,alpha,tau,Omega)
plt.title(title_str,size=16)


plt.tight_layout()

fig_name="lyapexp_vs_time-classical_coupled-gamma={0:0.4f}_alpha={1:0.2f}_tau={2:0.2f}_omega={3:0.2f}_omega={4:d}.pdf".format(gamma,alpha,tau,Omega,N_T)
plt.savefig(fig_name)

plt.show()



