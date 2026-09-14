#!/usr/bin/env python3
import numpy as np

def circular_slot_diff(a,b):
    d=abs(int(a)-int(b)); return min(d,48-d)

def corr_distance(a,b):
    return float(1-np.corrcoef(a,b)[0,1])

def bh(p):
    p=np.asarray(p,float); n=len(p); order=np.argsort(p); out=np.empty(n); prev=1.0
    for rank,idx in reversed(list(enumerate(order,start=1))):
        prev=min(prev,p[idx]*n/rank); out[idx]=prev
    return np.clip(out,0,1)
assert circular_slot_diff(1,48)==1
assert abs(corr_distance(np.arange(48),np.arange(48)))<1e-12
q=bh([.01,.02,.5]); assert len(q)==3 and np.all((q>=0)&(q<=1))
print("TEMPORAL DRIFT FEATURES SELFTEST: PASS")
