import numpy as np, sys
C2=1.4387769
par, parsum, blo, bhi = sys.argv[1], sys.argv[2], 9900., 14400.
cols=["CO2_7","CO2_8","CO2_9","CO2_10","CO2_11","CO2_12","CO2_13","CO2_14","CO2_121","CO2_15"]
hdr=open(parsum).readline().split(); rows=[]
for l in open(parsum).readlines()[1:]:
    w=l.split()
    if float(w[0])>1000: break
    rows.append([float(w[0])]+[float(w[hdr.index(c)]) for c in cols])
tab=np.array(rows)
QR={t:np.array([np.interp(296,tab[:,0],tab[:,j])/np.interp(t,tab[:,0],tab[:,j]) for j in range(1,11)]) for t in (230.,735.,750.)}
codes="1234567890"; L=[]
for l in open(par):
    v=float(l[3:15])
    if v<blo or v>=bhi or l[2] not in codes: continue
    L.append((v,float(l[15:25]),max(float(l[45:55]),0),codes.index(l[2])))
L=np.array(L); v,s,e,iso=L.T; iso=iso.astype(int)
sat=lambda t: s*QR[t][iso]*np.exp(-C2*e*(1/t-1/296))*(1-np.exp(-C2*v/t))/(1-np.exp(-C2*v/296))
nb=int((bhi-blo)/100); j=((v-blo)//100).astype(int)
out=np.column_stack([blo+100*np.arange(nb),np.bincount(j,minlength=nb),np.bincount(j,sat(735.),nb),np.bincount(j,sat(230.),nb)])
np.savetxt(sys.argv[3],out,header="bin_lo n S735 S230 (HITRAN2020 CO2, own-isotopologue Q)",fmt=["%.0f","%.0f","%.4e","%.4e"])
print(out[:,1].sum(),'lines; max nu',v.max())
