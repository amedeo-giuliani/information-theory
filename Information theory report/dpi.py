import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import commpy

def getProbabilities(sequence):
	freq = {}
	probs = []
	l = len(sequence)
	for item in sequence:
		if item in freq:
			freq[item] += 1
		else:
			freq[item] = 1
	for i in freq.values():
		probs.append(i/l)
	return probs

def getEntropy(pdf):
	return -sum(np.multiply(pdf,np.log2(pdf)))

def getJointEntropy(joint_pdf):
	return -sum(np.multiply(joint_pdf[joint_pdf!=0],np.log2(joint_pdf[joint_pdf!=0])))

def getJointDistribution(s1,s2,b):
	h,_,_ = np.histogram2d(s1,s2,bins=b)
	joint_pdf = h.flatten()/len(s1)
	return joint_pdf

midf = pd.DataFrame(columns=['idx','mi_xy','mi_xz'])
weights = [0.5,0.5]
epsilon1 = 0.3 # error rate of BSC 1
epsilon2 = 0.1 # error rate of BSC 2

for i in range(31):
    # generate input sequence and its statistics
	x = np.random.choice([0,1],200,p=weights)
	pdf_x = getProbabilities(x)
	h_x = getEntropy(pdf_x)
    
    # x goes through the first BSC
	y = commpy.bsc(x,epsilon1)
	# compute output statistics
	pdf_y = getProbabilities(y)
	h_y = getEntropy(pdf_y)
	pdf_xy = getJointDistribution(x,y,len(weights))
	h_xy = getJointEntropy(pdf_xy)

    # the output y is the input of another BSC
	z = commpy.bsc(y,epsilon2)
	# compute output statistics
	pdf_z = getProbabilities(z)
	h_z = getEntropy(pdf_z)
	pdf_xz = getJointDistribution(x,z,len(weights))
	h_xz = getJointEntropy(pdf_xz)

    # compute mutual informations with relationship (9) in the report
	mi_xy = h_x + h_y - h_xy
	mi_xz = h_x + h_z - h_xz
	# store them in a dataframe
	tmp = pd.DataFrame(columns=['idx','mi_xy','mi_xz'])
	tmp['mi_xy'] = [mi_xy]
	tmp['mi_xz'] = [mi_xz]
	tmp['idx'] = i
	midf = midf.append(tmp)

# plot results
plt.scatter(midf['idx'],midf['mi_xy'],facecolors='none',edgecolors='r')
plt.scatter(midf['idx'],midf['mi_xz'],facecolors='none',edgecolors='b')
plt.legend(['I(X;Y)','I(X;Z)'])
plt.savefig('mi.png',dpi=480)
