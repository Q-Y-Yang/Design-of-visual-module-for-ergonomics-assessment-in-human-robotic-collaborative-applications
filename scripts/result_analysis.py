import numpy as np
import csv
from matplotlib import pyplot as plt


#lookup tables for obtaining final risk level
def lookup(UL,WW,scoreB,F1,F2):
	tableA = np.array(([0,110,120,210,220,310,320,410,420],[11,1,2,2,2,2,3,3,3],[12,2,2,2,2,3,3,3,3],[13,2,3,3,3,3,3,4,4],[21,2,3,3,3,3,4,4,4],[22,3,3,3,3,3,4,4,4],[23,3,4,4,4,4,4,5,5],[31,3,3,4,4,4,4,5,5],[32,3,4,4,4,4,4,5,5],[33,4,4,4,4,4,5,5,5],[41,4,4,4,4,4,5,5,5],[42,4,4,4,4,4,5,5,5],[43,4,4,4,5,5,5,6,6],[51,5,5,5,5,5,6,6,7],[52,5,6,6,6,6,7,7,7],[53,6,6,6,7,7,7,7,8],[61,7,7,7,7,7,8,8,9],[62,8,8,8,8,8,9,9,9],[63,9,9,9,9,9,9,9,9]))

	#tableB = np.array(([0,11,12,21,22,31,32,41,42,51,52,61,62],[10,1,3,2,3,3,4,5,5,6,6,7,7],[20,2,3,2,3,4,5,5,5,6,7,7,7],[30,3,3,3,4,4,5,5,6,6,7,7,7],[40,5,5,5,6,6,7,7,7,7,7,8,8],[50,7,7,7,7,7,8,8,8,8,8,8,8],[60,8,8,8,8,8,8,8,9,9,9,9,9]))

	tableC = np.array(([0,10,20,30,40,50,60,70],[100,1,2,3,3,4,5,5],[200,2,2,3,4,4,5,5],[300,3,3,3,4,4,5,6],[400,3,3,3,4,5,6,6],[500,4,4,4,5,6,7,7],[600,4,4,5,6,6,7,7],[700,5,5,6,6,7,7,7],[800,5,5,6,7,7,7,7]))
	scoreA = tableA[np.argwhere(tableA == UL)[0,0], np.argwhere(tableA == WW)[0,1]]
	#scoreB = tableB[np.argwhere(tableB == N)[0,0], np.argwhere(tableB == TLE)[0,1]]
	#print(scoreA, scoreB)
	WA = 0
	NLT = 0
	WA = (F1 + scoreA)*100
	NTL = (F2 + scoreB)*10
	if WA>800:
		WA = 800
	if NTL>70:
		NTL = 70

	scoreC = tableC[np.argwhere(tableC == WA)[0,0], np.argwhere(tableC == NTL)[0,1]]
	
	return scoreC

#init
flag=0
upose=[]
lpose=[]
mocap_angles=[]
IK=[]
lIK=[]
uscore_pose=[]
uscore_IK = []
lscore_pose=[]
lscore_IK = []
scoreB=[]
WW=[]
sampleuIK=[]
samplelIK=[]
sampleupose=[]
samplelpose=[]
riskIK=[]
risk_pose=[]
diff=[]
riskdiff=[]


#data reading from .csv          
with open('new_IKuarm.csv') as k:  
    r = csv.reader(k)
    for i,row in enumerate(r):
        IK.append(float(row[0]))
        if float(row[0].replace('- ','')) <20:
            uscore_IK.append(1)
        elif float(row[0].replace('- ',''))>20 and  float(row[0].replace('- ',''))<45:
            uscore_IK.append(2)
        elif float(row[0].replace('- ',''))>45 and float(row[0].replace('- ',''))<90:
            uscore_IK.append(3)
        else: 
            uscore_IK.append(4)

with open('new_IKlarm.csv') as k:  
    r = csv.reader(k)
    for i,row in enumerate(r):
        lIK.append(float(row[0]))
        if float(row[0].replace('- ',''))>60 and  float(row[0].replace('- ',''))<100:
            lscore_IK.append(2)
        else: 
            lscore_IK.append(1)

with open('scores2204-2.csv') as s:	#uarm, larm, wrist score, and scoreB from openpose
	s = csv.reader(s)
	for i,row in enumerate(s):
		if row[0] == "data:":
			flag = i+1
		if i == flag and i!=0:
			upose.append(float(row[0].replace('- ','')))
			if float(row[0].replace('- ','')) <20:
				 uscore_pose.append(1)
			elif float(row[0].replace('- ',''))>20 and  float(row[0].replace('- ',''))<45:
				uscore_pose.append(2)
			elif float(row[0].replace('- ',''))>45 and float(row[0].replace('- ',''))<90:
				uscore_pose.append(3)
			else: 
				uscore_pose.append(4)
	
		if i == flag + 1 and i!=1:
			lpose.append(float(row[0].replace('- ','')))
			if float(row[0].replace('- ',''))>60 and  float(row[0].replace('- ',''))<100:
				lscore_pose.append(2)
			else: 
				lscore_pose.append(1)
		if i == flag + 2 and i!=2:
			WW.append(float(row[0].replace('- ','')))
		if i == flag + 3 and i!=3:
			scoreB.append(float(row[0].replace('- ','')))
		if i == flag + 4 and i!=4:
			risk_pose.append(float(row[0].replace('- ','')))



#sample IK frames
k=1
while k<8367:
    sampleuIK.append(uscore_IK[k])	#score of upper arm
    samplelIK.append(lscore_IK[k])	#score of lower arm
    sampleupose.append(IK[k])		#upper arm angular value
	samplelpose.append(lIK[k])
    k=k+20							#sample rate



F1=0	#ergonomics force scores, default 0
F2=0

p=0

#calculating risk levels of IK based on upper, lower arm angles, and other scores from openpose.
for p in range(0,min(len(sampleuIK),len(WW))):
    UL=int(sampleuIK[p])*10 + int(samplelIK[p])
    riskIK.append(lookup(UL,WW[p],scoreB[p],F1,F2))

#comparing final risk levels of the first 300 well-synchronized frames     
diff=abs(np.array(riskIK[:300]) - np.array(risk_pose[:300]))    
#count the number of non-different frames       
riskdiff.append(np.sum(diff==0))

print(len(riskdiff))
print(sum(riskdiff))
print(riskdiff)

j=1
error=0
#calculating errors based different risk classes
for j in range(0,300):
    if risk_pose[j]<3:
        if riskIK[j]<3:
            pass
        else:
            error = error+1
    elif risk_pose[j]<5 and risk_pose[j]>2:
        if riskIK[j]<5 and riskIK[j]>2:
            pass
        else:   error = error+1
    elif risk_pose[j]<7 and risk_pose[j]>4:
        if riskIK[j]<7 and riskIK[j]>4:
            pass
        else:   error = error+1
    elif risk_pose[j]==7:
        if riskIK[j]==7:
            pass
        else:   error = error+1
print(error)

frame_id=np.arange(0,404,1)

#plotting
plt.figure()
plt.ylabel('angular value')
plt.xlabel("frames")
plt.margins(x=0)
plt.plot(frame_id, sampleupose[:404], c='blue', label='IKUArm')#riskIK

plt.legend()
#plt.hlines(60,0,404, colors='c',linestyles='dashed')
#plt.hlines(100,0,404, colors='c',linestyles='dashed')
#plt.hlines(90,0,404, colors='c',linestyles='dashed')
plt.legend(loc='lower right')
plt.twiny()
plt.margins(x=0)
plt.plot(frame_id, upose, c='red', label='OpenPose_UArm')
plt.legend(loc='lower left')
##plt.plot(frame_id, sampleupose[:404],c='blue',label='IKLArm')
#plt.plot(frame_id,lpose,c='red',label='OpenPose_LArm')
plt.title('Arm Position')


#plt.ylim = (40,180)
#plt.yticks(np.arange(60,170,10))	

#ax=plt.gca()
#ax.xaxis.set_ticks_position('bottom')
#ax.yaxis.set_ticks_position('left')
#ax.spines['bottom'].set_position(('data',0))
#ax.spines['left'].set_position(('data',0))
plt.show()




        
