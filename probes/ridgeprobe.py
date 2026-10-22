"""Probe: random-direction rank-spline ("ridge spline") nuisance block vs the column-budget rule.
Design = budgeted additive rank spline + raw linear Z + m random directions a, each a 1-D cubic spline on
rank(a . normal_scores(Z)). No interaction block, no bandwidth. Total columns ~ n_train / 8. Monkeypatches
gfcm.core._design; the package itself is untouched."""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,'/Users/pavelaverin/Projects/gfcm'); sys.path.insert(0,'/Users/pavelaverin/Projects/gfcm/experiments')
import numpy as np, pandas as pd, dgp
from dataclasses import replace
from multiprocessing import Pool
from scipy.stats import norm, rankdata
from sklearn.preprocessing import SplineTransformer
from dagsampler import CausalDataGenerator
import gfcm.core as g
from gfcm import GFCMConfig, unified_test
N=2000; REPS=100; NTR=0.8*N; KR=8; _orig=g._design; _M=[0]
def _design_rr(Z,K,cfg,z_cat=None):
    base=_orig(Z,K,cfg,z_cat); Z=np.asarray(Z,float); d=Z.shape[1] if Z.ndim>1 else 1
    if _M[0]==0 or d<2: return base
    n=len(Z); G=norm.ppf(np.column_stack([rankdata(Z[:,j])/(n+1) for j in range(d)]))
    A=np.random.default_rng(2024).standard_normal((_M[0],d)); A/=np.linalg.norm(A,axis=1,keepdims=True)
    P=G@A.T; R=np.column_stack([rankdata(P[:,k])/(n+1) for k in range(_M[0])])
    return np.column_stack([base,SplineTransformer(n_knots=KR,degree=3).fit_transform(R)])
g._design=_design_rr
def kfit(d,m):   # additive knots that fit the n_train/8 budget next to d linear + m ridge splines
    return int(min(g.k_rule(N),max(4,(NTR/8-d-m*(KR+2))//d-2)))
ARMS={"default":(GFCMConfig(),0),"cols_1/8":(GFCMConfig(max_cols_frac=0.125),0),"ridge_m4":(GFCMConfig(inter=False),4),
      "ridge_m8":(GFCMConfig(inter=False),8),"ridge_m16":(GFCMConfig(inter=False),16)}
def data(fam,cell,dS,it):
    if fam=="inter": c=dgp.tab_interaction(N,70000+it)[cell]; zc=["Z1","Z2"]
    else: c=(dgp.calib_nulls if fam=="calib" else dgp.power_ladder)(N,70000+it,dS)[cell]; zc=[f"Z{i}" for i in range(1,dS+1)]
    c["simulation_params"]["store_ci_oracle"]=False
    df=CausalDataGenerator(c).simulate()["data"]; return df["X"].to_numpy(float),df["Y"].to_numpy(float),df[zc].to_numpy(float)
def job(a):
    fam,cell,dS,it=a; X,Y,Z=data(fam,cell,dS,it); d=Z.shape[1]; rows=[]
    for nm,(cfg,m) in ARMS.items():
        _M[0]=m; K=kfit(d,m) if m else None
        try:
            D=unified_test(X,Y,Z,config=cfg,K=K,detail=True)
            rows.append(dict(fam=fam,cell=cell,dS=d,it=it,arm=nm,p=D["acat3"],p_ss=D["acat4_ss"],p_tt=D["acat4_tt"]))
        except Exception as e: rows.append(dict(fam=fam,cell=cell,dS=d,it=it,arm=nm,err=str(e)[:80]))
    _M[0]=0; return rows
if __name__=="__main__":
    jobs=[("inter",c,2,it) for c in ("additive","interaction") for it in range(REPS)]
    jobs+=[(f,c,dS,it) for dS in (3,10,20) for f,cs in (("calib",["heavy_tail","hetero","nonlin_mean"]),("ladder",["null","scale","tail_shape"])) for c in cs for it in range(REPS)]
    with Pool(4) as P: R=[r for rs in P.imap_unordered(job,jobs,chunksize=8) for r in rs]
    d=pd.DataFrame(R); d.to_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)),"ridgeprobe.parquet"))
    if "err" in d and d.err.notna().any(): print(d[d.err.notna()].groupby(["arm","err"]).size())
    for col,name in (("p","DEFAULT BLOCKS"),("p_ss","+ SCALE-SCALE"),("p_tt","+ TAIL-TAIL")):
        t=d.assign(r=(d[col]<0.05).astype(float)).pivot_table(index=["dS","fam","cell"],columns="arm",values="r")[list(ARMS)]
        print(f"\n=== {name}: rejection at 0.05, n={N}, {REPS} reps (all cells are NULLS except ladder scale / tail_shape) ===\n",t.round(3).to_string())
    print("additive knots used:",{(d_,m):kfit(d_,m) for d_ in (2,3,10,20) for m in (4,8,16)})
