import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks
import datetime as dt
from distance_matrix import dm

############################################################################

def convert_date(date_bytes):
    return mdates.strpdate2num('%d-%m-%Y')(date_bytes.decode('ascii'))

days, discharge = np.loadtxt("Discharge.dat",converters={0: convert_date}, unpack=True)
Day, Q = np.loadtxt("Discharge.dat",converters={0: convert_date}, unpack=True)

############################################################################

plt.plot_date(x=Day, y=Q,fmt='r-',linewidth=1)
plt.xlabel("Year")
plt.ylabel("Discharge")
plt.tight_layout()
plt.savefig("Discharge.png",dpi=2000)
plt.show()



############################################################################

N=[]
for i in range(1,58037,365):
    X=discharge[i:i+365]
    n=[]
    while len(n)<=30:
        eps=90
        th=np.percentile(X,eps) # Threshold
        idx=(X>=th)
        X_new=X[idx]
        peaks,_=find_peaks(X_new)
        indx=[]
        for i in peaks:
            result=np.where(X_new[i]==X[:])
            indx+=list(result[-1])
        X[idx]=0 # Deleting the values which are above threshold
        n+=indx
    N.append(n)

N0=[]
N_count=[]
# arranging all the index for each year

from collections import OrderedDict
for i in range(160):
    N1=list(OrderedDict.fromkeys(N[i]))
    
    N0.append(N1)
    N_count+=list(N1)

Y=Q.copy()
t=np.arange(1,58439) # Regular time 

dis_Q=[] # discharge events in each year
T=[] # event time
for i in range(0,160):
    Q_new=Q[i*365:(i+1)*365]
    QQ=Q_new[N0[i]]
    dis_Q+=list(QQ)
    t_new=t[i*365:(i+1)*365]
    tt=t_new[N0[i]]
    T+=list(tt)

#############################################################################    

dis_Q=np.array(dis_Q) 
T=np.array(T)         
Dis_mat=dm(t,T,dis_Q,365) # Calling distance matrix function as dm
eps=np.percentile(Dis_mat,15) # using an optimal threshold 
j=np.where(Dis_mat<eps)
Rp=np.zeros(Dis_mat.shape, dtype="float")
Rp[j]=1.
Rp=np.delete(Rp,(-1),axis=0)
Rp=np.delete(Rp,(-1),axis=1)

#############################################################################
    
year_dt = []
for i in range(0,len(Day)):
    da=dt.datetime.fromordinal(int(Day[i]))
    if (da.month == 1) and (da.day == 1):
        year_dt.append(da.year)
year_dt = np.array(year_dt)
tx, ty = np.meshgrid(year_dt, year_dt)
plt.pcolormesh(tx, ty, Rp, cmap=plt.cm.gray_r)
plt.xlabel('Year')
plt.ylabel('Year')
plt.tight_layout()
plt.savefig("Recurrence plot.png",dpi=2000)
plt.show()























