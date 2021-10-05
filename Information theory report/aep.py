import numpy as np
import matplotlib.pyplot as plt

n = 5
k = 10000
N = 200
epsilon = 0.15
q = 0.25 #probability of 0 -> probability of 1 is 1-q

#compute the info of each symbol
zinfo = np.log2(1/q)
oinfo = np.log2(1/(1-q))
print("The info of value 0 is:",zinfo,"and the info of 1 is:",oinfo)

#entropy of binary rv with p = (q,1-q)
H = q*zinfo+(1-q)*oinfo
print("Entropy of Bernoulli RV with given q is",H)

#compute lower and upper bound for the avg info per sym
lb = H-epsilon
ub = H+epsilon
print("Lower bound is:",lb," and upper bound is:",ub)

prob = []

while n <= N:
	scount = 0
	for i in range(k):
		x = np.random.choice([0,1],n,p=[q,1-q]) #generate a sequence n symbols long
		ocount = np.count_nonzero(x) #count ones in sequence
		zcount = n - ocount #count zeros in sequence
		avginfo = (zcount*zinfo + ocount*oinfo)/n
		if avginfo < ub and avginfo > lb: #check if sequence is weakly epsilon-typical
			scount+=1
	prob.append(scount/k *100) #store the just computed typical set probability
	n += 5

#print(prob[0])
#print(prob[len(prob)-1])

#plot results
plt.plot(np.arange(5,N+5,5),prob)
plt.xlabel('Sequence length')
plt.ylabel('Probability of the typical set')
plt.savefig('aep.png',dpi=480)
