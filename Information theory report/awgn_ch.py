import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import commpy

K = np.arange(101)
noise_db = 0 #noise power in dB
var_z = 10**(noise_db/10) #calculate noise power in linear from dB
R = []
C_awgn = []

for k in K:
	x = np.random.normal(0,1,10000)
	#mu_x,sigma_x = stats.norm.fit(x)

	y = commpy.channels.awgn(x,noise_db)
	mu_y,sigma_y = stats.norm.fit(y) #estimate mean and stddev of y, which is Gaussian

	R.append(0.5 * np.log2(sigma_y**2 / var_z)) #compute and store actual info rate
	C_awgn.append(0.5 * np.log2(1 + 1/var_z)) #compute and capacity of awgn channel with given parameters

# compare R with C for K times	
plt.plot(K,C_awgn,'--')
plt.plot(K,R)
plt.legend(['Capacity','Actual rate'])
plt.ylim(0,1)
plt.grid()
plt.xlabel('Trials')
plt.ylabel('[bits]')
plt.title('Capacity vs actual rate with noise power 0 dB and P = 1')
plt.savefig('cap.png',dpi=480)
