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

np.set_printoptions(suppress=True,precision=4)


##### classical kicked top

# H(t) = \alpha S^x + \tau (S^z)^2 * (periodic kicks)

tau=7.0
alpha=1.5
Omega=2*np.pi
T=2*np.pi/Omega

############

"""

def EOMs(t,S,tau,alpha,Omega):
	''' S = [S, M] contains both the spin vector and its linearization. '''

	S_new=np.zeros_like(S)

	# evolve spin vector
	S_new[0] = -2.0*tau*S[1]*S[2]*np.cos(Omega*t)
	S_new[1] = -alpha*S[2] + 2.0*tau*S[0]*S[2]*np.cos(Omega*t)
	S_new[2] =  alpha*S[1]

	# evolve linearization
	Jac_new = np.array([    [0.0, -2.0*tau*S[2]*np.cos(Omega*t), -2.0*tau*S[1]*np.cos(Omega*t)],
							[2.0*tau*S[2]*np.cos(Omega*t), 0.0, -alpha + 2.0*tau*S[0]*np.cos(Omega*t)],
							[0.0, alpha, 0.0]
						])

	S_new[3:6] = Jac_new.dot(S[3:6])
	S_new[6:9] = Jac_new.dot(S[6:9])
	S_new[9:] = Jac_new.dot(S[9:])

	
	return S_new


EOM_cont_args=[tau,alpha,Omega]

S_0_1=np.array([0.23,0.0,0.69]) 
S_0_1/=np.linalg.norm(S_0_1)

S_0_2=np.array([0.23,1E-6,0.69]) 
S_0_2/=np.linalg.norm(S_0_2)


dS_0 = S_0_1 - S_0_2

S_0=np.concatenate([S_0_1,[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0] ])


time=Floquet_t_vec(Omega,100)  #np.linspace(0.0,20.0*T,101)
SS_t=evolve(S_0,time[0],time,EOMs,f_params=EOM_cont_args,real=True,iterate=True,atol=1E-14,rtol=1E-14)

# calculate linear evolution matrix
lyap_exp=np.zeros((3,time.len),dtype=np.float64)
for j,SS in enumerate(SS_t):

	M=SS[3:].reshape(3,3)
	#chi = np.linalg.eigvalsh(M.T.dot(M))
	#chi2 = np.linalg.svd(M, compute_uv=False)
	chi2 = sp.linalg.svd(M, compute_uv=False)

	if j>0:
		lyap_exp[:,j] = np.log(chi2)/(1.0*time[j])


plt.plot(time.vals,lyap_exp.T)
plt.plot(time.vals,np.sum(lyap_exp, axis=0) )
plt.show()


exit()


plt.plot( time/time.T, S_t[0,:] )
plt.plot( time/time.T, np.linalg.norm(S_t[:3,:],axis=0) )
plt.show()

plt.plot( time/time.T, S_t[3,:] )
plt.plot( time/time.T, np.linalg.norm(S_t[3:,:],axis=0) )
plt.show()

print(S_0)

"""

#####################

def EOM_max(t,S,tau,alpha,Omega):
	''' S = [S, M] contains both the spin vector and its linearization. '''

	S_new=np.zeros_like(S)

	# evolve spin vector
	S_new[0] = -2.0*tau*S[1]*S[2]*np.cos(Omega*t)
	S_new[1] = -alpha*S[2] + 2.0*tau*S[0]*S[2]*np.cos(Omega*t)
	S_new[2] =  alpha*S[1]

	# evolve linearization
	Jac_new = np.array([    [0.0, -2.0*tau*S[2]*np.cos(Omega*t), -2.0*tau*S[1]*np.cos(Omega*t)],
							[2.0*tau*S[2]*np.cos(Omega*t), 0.0, -alpha + 2.0*tau*S[0]*np.cos(Omega*t)],
							[0.0, alpha, 0.0]
						])

	S_new[3:6] = Jac_new.dot(S[3:6])
	
	return S_new


EOM_cont_args=[tau,alpha,Omega]


S_0_1=np.array([0.23,0.0,0.69]) 
S_0_1/=np.linalg.norm(S_0_1)

S_0_2=np.array([1.0,1.0,1.0])/np.sqrt(3.0) 

S_0=np.concatenate([S_0_1, S_0_2])


time=Floquet_t_vec(Omega,1000)  #np.linspace(0.0,20.0*T,101)
SS_t=evolve(S_0,time[0],time,EOM_max,f_params=EOM_cont_args,real=True,iterate=True,atol=1E-14,rtol=1E-14)

# calculate linear evolution
m=1
skip_steps=m*time.len_T
lyap_exp=np.zeros((time.N//m+1,),dtype=np.float64)
measure_times=np.zeros_like(lyap_exp)
k=0
for j,SS in enumerate(SS_t):

	S_lin=SS[3:]
	
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

print(SS)
		
plt.plot(measure_times,lyap_exp)
plt.show()
