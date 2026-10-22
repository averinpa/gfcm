import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,'/Users/pavelaverin/Projects/gfcm'); sys.path.insert(0,'/Users/pavelaverin/Projects/gfcm/experiments')
import numpy as np, pandas as pd, dgp, logging
from multiprocessing import Pool
from dagsampler import CausalDataGenerator
from gfcm import GFCMConfig, unified_test
CFGS={"default":(GFCMConfig(),None),"gcv1":(GFCMConfig(gcv=1),None),"noint":(GFCMConfig(inter=False),None),
      "K8_noint":(GFCMConfig(inter=False),8),"K8_noint_gcv1":(GFCMConfig(inter=False,gcv=1),8),"K5_noint_gcv1":(GFCMConfig(inter=False,gcv=1),5)}
N=2000; REPS=150
def job(a):
    fam,cell,dS,it=a
    c=(dgp.calib_nulls if fam=="calib" else dgp.power_ladder)(N,60000+it,dS)[cell]; c["simulation_params"]["store_ci_oracle"]=False
    df=CausalDataGenerator(c).simulate()["data"]; X=df["X"].to_numpy(float); Y=df["Y"].to_numpy(float); Z=df[[f"Z{i}" for i in range(1,dS+1)]].to_numpy(float)
    rows=[]
    for nm,(cfg,K) in CFGS.items():
        try:
            D=unified_test(X,Y,Z,config=cfg,K=K,detail=True)
            rows.append(dict(fam=fam,cell=cell,dS=dS,it=it,cfg=nm,p=D["acat3"],p_ss=D["acat4_ss"],p_tt=D["acat4_tt"]))
        except Exception as e: rows.append(dict(fam=fam,cell=cell,dS=dS,it=it,cfg=nm,err=str(e)[:80]))
    return rows
if __name__=="__main__":
    jobs=[(f,c,dS,it) for dS in (20,10) for f,cs in (("calib",["hetero","heavy_tail","nonlin_mean"]),("ladder",["null","scale","tail_shape","nonmonotone_z2"])) for c in cs for it in range(REPS)]
    with Pool(4) as P: R=[r for rs in P.imap_unordered(job,jobs,chunksize=8) for r in rs]
    d=pd.DataFrame(R); d.to_parquet(os.path.dirname(__file__)+"/fixprobe.parquet")
    if "err" in d: print(d[d.err.notna()].groupby(["cfg","err"]).size())
    for col,name in (("p","DEFAULT BLOCKS"),("p_ss","+ SCALE-SCALE"),("p_tt","+ TAIL-TAIL")):
        t=d.assign(r=(d[col]<0.05).astype(float)).pivot_table(index=["dS","fam","cell"],columns="cfg",values="r")[list(CFGS)]
        print(f"\n=== {name}: rejection rate at 0.05 (calib + ladder/null = size; other ladder cells = power), n={N}, {REPS} reps ===\n",t.round(3).to_string())
