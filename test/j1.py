import numpy as np
from numba import njit, vectorize, float64


Z1 = 1.46819706421238932572E1
Z2 = 4.92184563216946036703E1

RP = np.array([
    -8.99971225705559398224E8,
    4.52228297998194034323E11,
    -7.27494245221818276015E13,
    3.68295732863852883286E15,
])

RQ= np.array([
    # 1.00000000000000000000E0,
    6.20836478118054335476E2,  2.56987256757748830383E5,
    8.35146791431949253037E7,  2.21511595479792499675E10,
    4.74914122079991414898E12, 7.84369607876235854894E14,
    8.95222336184627338078E16, 5.32278620332680085395E18,
])

PP= np.array([
    7.62125616208173112003E-4, 7.31397056940917570436E-2,
    1.12719608129684925192E0,  5.11207951146807644818E0,
    8.42404590141772420927E0,  5.21451598682361504063E0,
    1.00000000000000000254E0,
])
PQ= np.array([
    5.71323128072548699714E-4, 6.88455908754495404082E-2,
    1.10514232634061696926E0,  5.07386386128601488557E0,
    8.39985554327604159757E0,  5.20982848682361821619E0,
    9.99999999999999997461E-1,
])

QP= np.array([
    5.10862594750176621635E-2, 4.98213872951233449420E0,
    7.58238284132545283818E1,  3.66779609360150777800E2,
    7.10856304998926107277E2,  5.97489612400613639965E2,
    2.11688757100572135698E2,  2.52070205858023719784E1,
])
QQ= np.array([
    # 1.00000000000000000000E0,
    7.42373277035675149943E1, 1.05644886038262816351E3,
    4.98641058337653607651E3, 9.56231892404756170795E3,
    7.99704160447350683650E3, 2.82619278517639096600E3,
    3.36093607810698293419E2,
])

TWOOPI = 2*np.pi
THPIO4 = 3*np.pi/4
SQ2OPI = .79788456080286535588
PI04 = .7853981633974483096157


#  5.783185962946784521175995758455807035071 
DR1 = 5.783185962946784521175995758455807035071 
# 30.47126234366208639907816317502275584842 
DR2 = 30.47126234366208639907816317502275584842 


@njit(cache=True)
def polevl(x, coef, N):

    ans = coef[0]
    for i in range(1, N+1):
        ans = ans * x + coef[i]
    return ans

@njit(cache=True)
def p1evl(x, coef, N):

    ans = x + coef[0]
    for i in range(1, N):
        ans = ans * x + coef[i]
    return ans


@vectorize([float64(float64)])
def j0(x):
    x = np.abs(x)
    
    if x <= 5.:
        z = x * x
        if(x < 1e-5):
            return 1. - z / 4.
        
        p = (z-DR1) * (z - DR2)
        p = p * polevl(z, RP, 3) / p1evl(z, RQ, 8)
        return p

    w = 5. / x
    q = 25.0/(x*x)
    p = polevl( q, PP, 6) / polevl( q, PQ, 6 )
    q = polevl( q, QP, 7) / p1evl( q, QQ, 7 )
    
    xn = x - PI04
    p = p * np.cos(xn) - w * q * np.sin(xn)
    
    return p * SQ2OPI / np.sqrt(x)
        
@vectorize([float64(float64)])
def j1(x):
    w = np.abs(x)
    
    if w <= 5:
        z = x*x
        w = polevl(z, RP, 3)  / p1evl(z, RQ, 8)
        w = w * x * (z - Z1) * (z - Z2)
        return w

    w = 5.0 / x
    z = w * w
    p = polevl(z, PP, 6) / polevl(z, PQ, 6)
    q = polevl(z, QP, 7) / p1evl(z, QQ, 7)
    
    xn = x - THPIO4
    p = p * np.cos(xn) - w * q * np.sin(xn)
    return (p * SQ2OPI / np.sqrt(x))



if __name__ == "__main__":
    from scipy.special import j1 as BesselJ1
    from scipy.special import j0 as BesselJ0
    import matplotlib.pyplot as plt  
    
    x = np.linspace(-10, 10, 1000)
    plt.plot(x, j0(x), label='j0')
    plt.plot(x, BesselJ0(x), label='BesselJ0')
    plt.legend()
    plt.show()
    exit()
    import timeit
    print(timeit.timeit(lambda: BesselJ1(np.random.rand(100)), number=10000))
    print(timeit.timeit(lambda: j1(np.random.rand(100)), number=10000))
    
    x = np.random.rand(100)
    print(np.allclose(j1(x), BesselJ1(x)))