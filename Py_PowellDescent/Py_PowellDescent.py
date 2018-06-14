## Module Py_PowellDescent
'''
xMin,nCyc = powell(F,x,h=0.1,tol=1.0e-6)
Powell’s method of minimizing supplied function function F(x). With discarding of vector
which results in largest decrease

x = starting point
h = initial search increment used in Py_GoldSearch.bracket
xMin = mimimum point
nCyc = number of cycles
'''

from numpy import identity, array, dot, zeros, float64, argmax
from Py_GoldSearch import *
from math import sqrt

def Powell(F,x,h=0.001,tol=1.0e-6):

    def f(s): return F(x+s*v)

    n = len(x)
    df = zeros((n),dtype=float64)
    u = identity(n)*1.0

    for j in range(100):
        xOld = x.copy()
        fOld = F(xOld)

        for i in range(n):
            v = u[i]
            a,b = bracket(f,0.0,h)
            s, fMin = search(f,a,b)
            df[i] = fOld - fMin
            fOld = fMin
            x = x + s*v

        v = x - xOld
        a,b = bracket(f,0.0,h)
        s, fLast = search(f,a,b)
        x = x + s*v

        tolcheck = sqrt(dot(x-xOld,x-xOld)/n)
        if  tolcheck < tol: return x,j+1

        #discard direction of greatest descent
        #find the indicies of the maximum then discard and re-arrange the array
        iMax = int(argmax(df))              
        for i in range(iMax, n-1):
            u[i] = u[i+1]
            u[n-1] = v
        print("Powell did not converge")



