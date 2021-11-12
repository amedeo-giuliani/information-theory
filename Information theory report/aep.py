import numpy as np
import matplotlib.pyplot as plt

k = 10000
epsilon_vec = [0.05,0.1,0.2,0.3]
q_vec = [0.1,0.2,0.3,0.5] #probability of 0 -> probability of 1 is 1-q

for epsilon in epsilon_vec:
    n = 5
    q = q_vec[1]

    #compute the avg info per sym for each sequence x[i]
    zinfo = np.log2(1/q)
    oinfo = np.log2(1/(1-q))
    print("The info of value 0 is:",zinfo,"and the info of 1 is:",oinfo)

    #entropy of RV bernoulli(q)
    H = q*zinfo+(1-q)*oinfo
    print("Entropy of Bernoulli RV with given q is",H)

    #compute lower and upper bound for the avg info per sym
    lb = H-epsilon
    ub = H+epsilon
    print("Lower bound is:",lb," and upper bound is:",ub)

    prob = []

    while n <= 200:
        scount = 0
        for i in range(k):
            x = np.random.choice([0,1],n,p=[q,1-q]) #generate k sequences each n symbols long
            ocount = np.count_nonzero(x) #count ones in sequence
            zcount = n - ocount #count zeros in sequence
            avginfo = (zcount*zinfo + ocount*oinfo)/n
            if avginfo < ub and avginfo > lb: #check if sequence is weakly epsilon-typical
                scount+=1
        #print("Average info per symbol of the k sequences:",avginfo)
        #print("Epsilon-typical sequences:",scount,"/",k)
        #print("Probability of sequence belonging to the typical set:",scount/k *100,"%")
        prob.append(scount/k *100)
        n += 5

    print(prob[0])
    print(prob[len(prob)-1])
    plt.plot(np.arange(5,205,5),prob)
    plt.xlabel('Sequence length')
    plt.ylabel('Probability of the typical set')

plt.title('With q='+str(q_vec[1]))
plt.legend(['epsilon=0.05','epsilon=0.1','epsilon=0.2','epsilon=0.3'])
plt.savefig('aep1.png',dpi=480)

plt.clf()

for q in q_vec:
    n = 5
    epsilon = epsilon_vec[1]

    #compute the avg info per sym for each sequence x[i]
    zinfo = np.log2(1/q)
    oinfo = np.log2(1/(1-q))
    print("The info of value 0 is:",zinfo,"and the info of 1 is:",oinfo)

    #entropy of RV bernoulli(q)
    H = q*zinfo+(1-q)*oinfo
    print("Entropy of Bernoulli RV with given q is",H)

    #compute lower and upper bound for the avg info per sym
    lb = H-epsilon
    ub = H+epsilon
    print("Lower bound is:",lb," and upper bound is:",ub)

    prob = []

    while n <= 200:
        scount = 0
        for i in range(k):
            x = np.random.choice([0,1],n,p=[q,1-q]) #generate k sequences each n symbols long
            ocount = np.count_nonzero(x) #count ones in sequence
            zcount = n - ocount #count zeros in sequence
            avginfo = (zcount*zinfo + ocount*oinfo)/n
            if avginfo < ub and avginfo > lb: #check if sequence is weakly epsilon-typical
                scount+=1
        #print("Average info per symbol of the k sequences:",avginfo)
        #print("Epsilon-typical sequences:",scount,"/",k)
        #print("Probability of sequence belonging to the typical set:",scount/k *100,"%")
        prob.append(scount/k *100)
        n += 5

    print(prob[0])
    print(prob[len(prob)-1])
    plt.plot(np.arange(5,205,5),prob)
    plt.xlabel('Sequence length')
    plt.ylabel('Probability of the typical set')

plt.title('With epsilon='+str(epsilon_vec[1]))
plt.legend(['q=0.1','q=0.2','q=0.3','q=0.5'])
plt.savefig('aep2.png',dpi=480)
