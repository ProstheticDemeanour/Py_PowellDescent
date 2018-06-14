## module GoldSearch

''' a,b = bracket (f,xStart, h)
Finds the brackets (a,b) of a minimum point of the
user supplied scalar function f(x).
The search starts dfownhill from xStart with a step Length h.

x ,fMin = search(f, a, b, tol=1.0e-6)
Golden section method for determining x that minimsed the use supplied scalar function f(x)
The minimum must be bracketed in (a,b)

'''
from math import log, ceil

def bracket(f, x1, h):
    c = 1.618033989
    f1 = f(x1)
    x2 = x1 + h; f2 = f(x2)

    #determine downhill direction and change sign of h if needed
    if f2 > f1:
        h = -h
        x2 = x2 + h; f2 = f(x2)
        # check if minimum is between x1 - h and x1 + h
        if f2 > f1: return x2, x1 - h

    # search Loop
    for i in range (100):
        h= c*h
        x3 = x2 + h; f3 = f(x3)
        if f3 > f2: return x1,x3
        x1 = x2; x2 = x3
        f1 = f2; f2 = f3
        print("Bracket did not find minimum")

def search(f, a, b, tol=1.0e-9):
    nIter = 2*ceil(-2.078087*log(tol/abs(b-a)))
    R = 0.618033989
    C = 1.0 - R
    # First Telescoping
    x1 = R*a + C*b; x2 = C*a + R*b
    f1 = f(x1); f2 = f(x2)

    #Main Loop

    for i in range(nIter):
        if f1 > f2:
            a = x1
            x1 = x2; f1 = f2
            x2= C*a + R*b; f2 - f(x2)
        else:
            b = x2
            x2 = x1; f2 = f1
            x1 = R*a + C*b; f1 = f(x1)
    if f1 < f2: return x1, f1
    else: return x2, f2


