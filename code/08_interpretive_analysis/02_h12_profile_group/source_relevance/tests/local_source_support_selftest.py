#!/usr/bin/env python3
import numpy as np

def safe_relative(a,d):
    a=np.asarray(a,float); d=np.asarray(d,float); out=np.full(np.broadcast_shapes(a.shape,d.shape),np.nan); np.divide(100*a,d,out=out,where=np.abs(d)>1e-12); return out
D=np.array([2.,3.]); T=np.array([2.5,2.5]); L=np.array([4.,4.]); F=np.array([1.5,3.5])
ST=L-D; FT=D-T; FD=L-F; DVF=F-D; FTL=L-T; FVT=T-F
assert np.allclose(FTL,ST+FT); assert np.allclose(DVF,ST-FD); assert np.allclose(FVT,FD-FTL)
r=safe_relative(np.array([1.,1.]),np.array([2.,0.])); assert np.isclose(r[0],50.) and np.isnan(r[1])
print("LOCAL SOURCE SUPPORT SELFTEST: PASS")
