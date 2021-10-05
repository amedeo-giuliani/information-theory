import numpy as np
from matplotlib import pyplot as plt

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
	return -sum(np.multiply(joint_pdf[joint_pdf!=0],np.log2(joint_pdf[joint_pdf!=0]))) #ignore zero as they're useless and give also division by zero error

def getJointDistribution(s1,s2,b):
	if b != 0:
		h,_,_ = np.histogram2d(s1,s2,bins=b) #ignore every output but bin count
	else:
		h,_,_ = np.histogram2d(s1,s2) #ignore every output but bin count
	joint_pdf = h.flatten()/len(s1)
	return joint_pdf

def getKLD(p,q):
	return sum(np.multiply(p,np.log2(np.divide(p,q))))

weights = [0.5,0.5]
x = np.random.choice([0,1],200,p=weights)
pdf_x = getProbabilities(x)
h_x = getEntropy(pdf_x)
print('H(x)=%.2f'%h_x)

y = 1-x
pdf_y = np.subtract(1,pdf_x)
h_y = getEntropy(pdf_y)
print('H(y)=%.2f'%h_y)
pdf_xy = getJointDistribution(x,y,len(weights))
h_xy = getJointEntropy(pdf_xy)
print('H(x,y)=%.2f'%h_xy)

z = np.random.choice([0,1],200,p=weights)
pdf_z = getProbabilities(z)
h_z = getEntropy(pdf_z)
print('H(z)=%.2f'%h_z)
pdf_xz = getJointDistribution(x,z,len(weights))
h_xz = getJointEntropy(pdf_xz)
print('H(x,z)=%.2f'%h_xz)

print('D(x||y)=%.2f'%getKLD(pdf_x,pdf_y))
print('D(x||z)=%.2f'%getKLD(pdf_x,pdf_z))

pdf_x = np.array([pdf_x])
pdf_y = np.array([pdf_y])
pdf_z = np.array([pdf_z])
# compute p(x,y) = p(x)p(y), i.e. as they were independent
pdf_xy1 = np.matmul(pdf_x.T,pdf_y).flatten()
# compute p(x,z) = p(x)p(z), i.e. as they were independent (they actually are, though)
pdf_xz1 = np.matmul(pdf_x.T,pdf_z).flatten()
# I(x;z) = D(p(x,z)||p(x)p(z)
mi_xz = getKLD(pdf_xz,pdf_xz1)
pdf_xy[pdf_xy==0] = 1e-6 # to not get division by zero error
# I(x;y) = D(p(x,y)||p(x)p(y)
mi_xy = getKLD(pdf_xy,pdf_xy1)

print('I(x;y)=%.2f'%mi_xy)
print('I(x;z)=%.2f'%mi_xz)
