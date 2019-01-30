import numpy as np
import math

def mean(data):
    return sum(data)/len(data)

def getFeature(dataset):
    NLIQ,OCQ,HINDEX,IC = [],[],[],[]
    for j in dataset:
        j=j.split(",")[1:]
        NLIQ.append(float(j[3]))
        OCQ.append(float(j[4]))
        HINDEX.append(float(j[5]))
        IC.append(float(j[6]))
    features=[]
    for i in range(len(NLIQ)):
        features.append([NLIQ[i],OCQ[i],HINDEX[i],IC[i]])
    return features

def getData():
    NatJournals=open("Data/FV_classifier_national.csv", "r").readlines()
    IntJournals=open("Data/FV_classifier_Inter.csv", "r").readlines()
    NatJournals=NatJournals[1:]
    IntJournals=IntJournals[1:]
    np.random.shuffle(NatJournals)
    np.random.shuffle(IntJournals)
    NatTrain=getFeature(NatJournals[:15])
    IntTrain=getFeature(IntJournals[:15])
    NatTest=NatJournals[15:]
    IntTest=IntJournals[15:]
    TestData=NatTest+IntTest
    TestFeatures = getFeature(TestData)
    return TestFeatures,NatTrain,IntTrain

class MVNorm:
    def __init__(self,dataset):
        self.mu = []
        self.cvmatrix=[[0]*4]*4
        param=[]
        for i in range(len(dataset[0])):
            temp=[]
            for j in range(len(dataset)):
                 temp.append(dataset[j][i])
            self.mu.append(mean(temp))
            param.append(temp)
        self.cvmatrix=np.cov(param)
        
    def p(self,x):
        denom = 1/math.pow((2*math.pi),2)*math.pow(np.linalg.det(self.cvmatrix),0.5)
        power = (-0.5)*np.matmul(np.matmul(np.transpose(np.subtract(x,self.mu)),np.linalg.inv(self.cvmatrix)),np.subtract(x,self.mu))
        py_x = math.exp(power)*denom*(0.5/(1/12))
        descrim = power - (2*math.log(2*math.pi))-0.5*math.log(np.linalg.det(self.cvmatrix)) + math.log(0.5)
        return py_x,descrim



acc=[]
acc_nat=[]
acc_int=[]
risks=[]
miss=0
false_pos=0
wrong_class=0
bounds_66=[]
bounds_5=[]
overbound_5=0
overbound_66=0
no_risk=0

def errbound(beta,n1,n2):
    k1=np.matmul(np.multiply(beta*(1-beta)/2,np.transpose(np.subtract(n1.mu,n2.mu))),np.matmul(np.linalg.inv(np.add(np.multiply(beta,n1.cvmatrix),np.multiply(1-beta,n2.cvmatrix))),np.subtract(n1.mu,n2.mu)))
    k2 = 0.5*math.log(np.linalg.det(np.add(np.multiply(beta,n1.cvmatrix),np.multiply(1-beta,n2.cvmatrix)))/(math.pow(np.linalg.det(n1.cvmatrix),beta)*math.pow(np.linalg.det(n2.cvmatrix),1-beta)))
    #print(k1+k2)
    return 0.5 * math.exp(-(k1+k2))

verbose = int(input("Enter 1 to show each test case's results, 0 to show only final results:"))
if not verbose:
    print("Running test data through model....")

for i in range(10000):
    TestF, NatF, IntF =getData()
    NationalTrainer = MVNorm(NatF)
    InternationalTrainer = MVNorm(IntF)
    actual_class=[0]*5+[1]*7
    res=[]
    score=0
    score_nat=0
    score_int=0
    bounds_5.append(errbound(0.5,NationalTrainer,InternationalTrainer))
    bounds_66.append(errbound(1/3,NationalTrainer,InternationalTrainer))
    for X in TestF:

        y1,g1 = NationalTrainer.p(X)
        y2,g2 = InternationalTrainer.p(X)
        
        if g1>g2:
            res.append(0)
            risks.append(y2)
            if y2==0:
                no_risk+=1
                
        else:
            res.append(1)
            risks.append(y1)
            if y1==0:
                no_risk+=1
            
    for j in range(len(res)):
        if res[j]==actual_class[j]:
            score+=1
            if j>4:
                score_int+=1
            else:
                score_nat+=1
        else:
            wrong_class+=1
            if res[j]==1 and actual_class[j]==0:
                false_pos+=1
            if res[j]==0 and actual_class[j]==1:
                miss+=1
                
    if risks[i]>bounds_5[i]:
        overbound_5+=1
    if risks[i]>bounds_66[i]:
        overbound_66+=1
    
    if verbose:
        print(res,score)

    acc.append(score/12)
    acc_nat.append(score_nat/5)
    acc_int.append(score_int/7)

print()
print("Results:")
print("Overall accuracy-")
print("Average accuracy: ",mean(acc)," Highest Accuracy: ",max(acc)," Lowest Accuracy: ",min(acc))
print()
print("Classwise accuracy-")
print("Average accuracy for national class: ",mean(acc_nat),"Average accuracy for International class: ",mean(acc_int))
print()
print("Average risk: ",mean(risks)," Max Risk :", max(risks)," Min Risk :",min(risks))
print()
print(wrong_class,"missclassifications over",10000*12,"classifications, out of which", false_pos, "were false positive and",miss,"were misses")
print()
print("As per Bhattacharya bounds, Minimum error upper bound:",min(bounds_5),"Maximum error upper bound:",max(bounds_5),"Average error upper bound:",mean(bounds_5))
print("As per Chernoff bounds at beta = 1/3 , Minimum error upper bound:",min(bounds_66),"Maximum error upper bound:",max(bounds_66),"Average error upper bound:",mean(bounds_66))
