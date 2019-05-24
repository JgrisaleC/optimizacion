import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)


def plot_fun(f_name, points=None, limits=None, params=None):
    limits_dict = {'ackley': (-5, 5, -5, 5),
                   'bukin': (-15, -5, -3, 3),
                   'cross_in_tray': (-5, 5, -5, 5),
                   'drop_wave': (-5.2, 5.2, -5.2, 5.2),
                   'eggholder': (-520, 520, -520, 520),
                   'griewank': (-5, 5, -5, 5),
                   'holder_table': (-10, 10, -10, 10),
                   'langermann': (0,10, 0, 10),
                   'levy': (-10, 10, -10, 10),
                   'bochavesky': (-100, 100, -100, 100),
                   'perm_zero_d_beta': (-2, 2, -2, 2),
                   'rot_hyper_ellipsoid': (-100, 100, -100, 100),
                   'sphere': (-100, 100, -100, 100),
                   'sum_powers': (-1, 1, -1, 1),
                   'sum_squares': (-10, 10, -10, 10),
                   'trid': (-4, 4, -4, 4),
                   'booth': (-10, 10, -10, 10),
                   'matyas': (-10, 10, -10, 10),
                   'mccormick': (-1.5, 4, -3, 4),
                   'power_sum': (-2, 2, -2, 2),
                   'zakharov': (-5, 10, -5, 10),
                   'three_hump_camel': (-5, 5, -5, 5),
                   'six_hump_camel': (-1.5, 1.5, -1.5, 1.5),
                   'dixon_price': (-10, 10, -10, 10),
                   'rosenbrock': (-2.048, 2.048, -2.048, 2.048),
                   'de_jong_5': (-50, 50, -50, 50),
                   'easom': (-20, 20, -20, 20),
                   'michalewicz': (0, 4, 0, 4),
                   'beale': (-4.5, 4.5, -4.5, 4.5),
                   'branin': (-5, 15, -5, 15),
                   'goldstein_price': (-2, 2, -2, 2),
                   'perm_d_beta': (-3, 3, -3, 3),
                   'styblinski_tang': (-5, 5, -5, 5),
                   'parsopoulos': (-5, 5, -5, 5)
                   }
    #
    # params_dict = {ackley: ()
    #
    # }
    if limits is None:
        limits = limits_dict[f_name.__name__]
    x = np.linspace(limits[0], limits[1], 1000)
    y = np.linspace(limits[2], limits[3], 1000)
    X, Y = np.meshgrid(x, y)
    Z = f_name((X, Y))
    data = [go.Surface(x=x, y=y, z=Z)]
    if points is not None:
        for p in points:
            data.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode='markers'))
    fig = go.Figure(data=data)
    plotly.offline.plot(fig)


# -------------------------------------MANY LOCAL MINIMA-------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def ackley(x, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    sum_1 = np.sqrt(1/d*sum(x[i]**2 for i in range(len(x))))
    sum_2 = sum(np.cos(c*x[i]) for i in range(len(x)))
    return -a*np.exp(-b*sum_1) - np.exp(1/d*sum_2) + a + np.exp(1)


def bukin(x):
    """
    minimiser: x = (-10,1)
    """
    return 100*np.sqrt(np.abs(x[1]-0.01*(x[0]**2)))+0.01*np.abs(x[0]+10)


def cross_in_tray(x):
    """
    minimisers: x = (1.3491, -1.3491), (1.3491, 1.3491), (-1.3491, 1.3491), (-1.3491, -1.3491)
    """
    return -1e-4*((np.abs(np.sin(x[0])*np.sin(x[1])*np.exp(abs(100-np.sqrt(x[0]**2+x[1]**2)/np.pi)))+1)**0.1)


def drop_wave(x):
    """
    minimiser: x = (0,0)
    """
    top = 1 + np.cos(12*np.sqrt(x[0]**2+x[1]**2))
    bot = 0.5*(x[0]**2 + x[1]**2)+2
    return -top/bot


def eggholder(x):
    """
    minimiser: x = (512, 404.2319)
    """
    return -(x[1]+47)*np.sin(np.sqrt(np.abs(x[1] + x[0]/2 + 47))) - x[0]*np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))


def griewank(x, d=2):
    """
    minimiser : x = 0
    """
    s = sum((x[i]**2)/4000 for i in range(d))
    p = 1
    for i in range(d):
        p *= np.cos(x[i]/np.sqrt(i+1))
    # p = np.prod(np.array([np.cos(x[i]/np.sqrt(i+1)) for i in range(d)]))
    return s - p + 1


def holder_table(x):
    """
    minimisers : x = (8.05502, 9.66459), (8.05502, -9.66459), (-8.05502, 9.66459), (-8.05502, -9.66459)
    """
    return -np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(abs(1-np.sqrt(x[0]**2+x[1]**2)/np.pi)))


def langermann(x, d=2, m=5, c=(1, 2, 5, 2, 3), a=np.array([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]]).T):
    """
    minimiser: x= (2.00299219, 1.006096)
    """
    return sum(c[i] * np.exp(-1 / np.pi * sum((x[j] - a[i, j]) ** 2 for j in range(d))) *
               np.cos(np.pi * sum((x[j] - a[i, j]) ** 2 for j in range(d))) for i in range(m))


def levy(x, d=2):
    """
    minimiser: x = (1,...,1)
    d is the dimension of x
    """
    w = lambda y: 1 + (y-1)/4
    s = sum((w(x[i])-1)**2*(1+10*np.sin(np.pi*w(x[i])+1)**2) for i in range(1,d-1))
    return np.sin(np.pi*w(x[0]))**2 + s + (w(x[-1])-1)**2*(1+np.sin(2*np.pi*w(x[-1]))**2)

# -------------------------------------BOWL-SHAPED-------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def bochavesky(x, n=1):
    """
    minimiser: x = 0
    n is the number of the bochavesky function
    """
    if n == 1:
        return x[0]**2+2*(x[1]**2)-0.3*np.cos(3*np.pi*x[0])-0.4*np.cos(4*np.pi*x[1])+0.7
    elif n == 2:
        return x[0]**2+2*(x[1]**2)-0.3*np.cos(3*np.pi*x[0])*np.cos(4*np.pi*x[1])+0.3
    elif n == 3:
        return x[0]**2+2*(x[1]**2)-0.3*np.cos(3*np.pi*x[0]+4*np.pi*x[1])+0.3
    else:
        print('argument n is invalid, n must be 1, 2 or 3')
        raise ValueError


def perm_zero_d_beta(x, d=2, beta=1):
    """
    minimiser: (1, 1/2, ..., 1/d)
    d: number of dimension of x
    """
    return sum(sum((j+beta)*(x[j-1]**i-1/(j**i)) for j in range(1, d+1))**2 for i in range(1,d+1))


def rot_hyper_ellipsoid(x, d=2):
    """
    minimiser: x = 0
    d is the dimension of x
    """
    return sum(sum(x[j]**2 for j in range(i+1)) for i in range(d))


def sphere(x, d=2):
    """
    minimiser: x = 0
    d is the dimension of x
    """
    return sum(x[i]**2 for i in range(d))


def sum_powers(x, d=2):
    """
    minimiser: x = 0
    d is the dimension of x
    """
    return sum(np.abs(x[i])**(i+2) for i in range(d))


def sum_squares(x, d=2):
    """
    minimiser: x = 0
    d is the dimension of x
    """
    return sum((i+1)*(x[i]**2) for i in range(d))


def trid(x, d=2):
    """
    minimiser: x_i = i(d+1-i) for all i=1,2,...,d
    d is the dimension of x
    """
    s1 = sum((x[i]-1)**2 for i in range(d))
    s2 = sum(x[i]*x[i-1] for i in range(1, d-1))
    return s1 - s2


# -------------------------------------PLATE-SHAPED------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

def booth(x):
    """
    minimiser: x= (1,3)
    """
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def matyas(x):
    """
    minimiser: x= (0,0)
    """
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]


def mccormick(x):
    """
    minimiser: x= (-0.54719, -1.54719)
    """
    return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1


def power_sum(x, d=2, b=(8, 18)):
    return sum((sum(x[j]**(i+1) for j in range(d))-b[i])**2 for i in range(d))


def zakharov(x, d=2):
    """
    minimiser: x = 0
    d is the dimension of x
    """
    s1 = sum(x[i]**2 for i in range(d))
    s2 = sum(0.5*(i+1)*x[i] for i in range(d))**2
    s3 = sum(0.5*(i+1)*x[i] for i in range(d))**4
    return s1 + s2 + s3


# -------------------------------------VALLEY-SHAPED-----------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def three_hump_camel(x):
    """
    global minimiser: x= (0,0)
    Has three local minima
    """
    return 2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2


def six_hump_camel(x):
    """
    global minimisers : x = (0.0898, -0.7126)  and  x = (-0.0898, 0.7126)
    Has six local minima
    """
    return (4 - 2.1*x[0]**2 + (x[0]**4)/3)*(x[0]**2) + x[0]*x[1] + (-4 + 4*x[1]**2)*(x[1]**2)


def dixon_price(x, d=2):
    """
    minimiser (for d=2): x = (1,1/sqrt(2))
    d is the dimension of x
    """
    s = sum((i+1)*(2*x[i]**2-x[i-1])**2 for i in range(1,d))
    return (x[0] - 1)**2 + s


def rosenbrock(x, d=2):
    """
    minimiser : x = (1,..., 1)
    d is the dimension of x
    """
    return sum(100*(x[i+1]-x[i]**2)**2+(x[i]-1)**2 for i in range(d-1))


# -------------------------------------STEEP RIDGES/DROPS------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def de_jong_5(x):
    r1 = [-32, -16, 0, 16, 32]*5
    r2 = [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5
    a = np.array([r1, r2])
    return 1/(0.002 + sum(1/(i+1 + (x[0]-a[0, i])**6 + (x[1]-a[1,i])**6) for i in range(25)))


def easom(x):
    """
    minimiser : x = (pi,pi)
    """
    return -np.cos(x[0])*np.cos(x[1])*np.exp(-(x[0]-np.pi)**2-(x[1]-np.pi)**2)


def michalewicz(x, d=2, m=10):
    """
    d is the dimension of x
    m defines steepness of valleys and ridges
    Has d! local minima
    """
    return -sum(np.sin(x[i])*np.sin((i+1)*(x[i]**2)/np.pi)**(2*m) for i in range(d))


# -------------------------------------OTHERS------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def beale(x):
    """
    minimiser : x = (3,0.5)
    """
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2


def branin(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    """
    minimisers : x = (-pi,12.275) , (pi, 2.275), (9.42478, 2.475)
    """
    return a*(x[1] - b*x[0] + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s


def colville(x):
    """
    minimiser : x = (1,1,1,1)
    """
    s1 = 100*(x[0]**2-x[1])**2 + (x[0]-1)**2 + (x[2]-1)**2 + 90*(x[2]**2-x[3])**2
    s2 = 10.1*((x[1]-1)**2 + (x[3]-1)**2) + 19.8*(x[1]-1)*(x[3]-1)
    return  s1 + s2


def goldstein_price(x):
    """
    minimiser : x = (0,-1)
    """
    s1 = 1 + (x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2)
    s2 = 30 + (2*x[0] - 3*x[1])**2*(18- 32*x[0] + 12*x[1]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
    return s1*s2


def perm_d_beta(x, d=2, beta=1):
    """
    minimiser : x = (1,2,...,d)
    """
    return sum(sum(((j+1)**(i+1)+beta)*((x[j]/(j+1))**(i+1)-1) for j in range(d))**2 for i in range(d))


def styblinski_tang(x, d=2):
    """
    minimiser : x = (-2.903534,...,-2.903534)
    """
    return 0.5*sum(x[i]**4 - 16*x[i]**2 + 5*x[i] for i in range(d))


def parsopoulos(x):
    """
    infinite global minimisers: x = (k*(pi/2), l*pi) for k=1,2,3,... and l= 0, 1, 2, ...
    """
    return np.cos(x[0])**2 + np.sin(x[1])**2


# plot_fun(langermann, points=[np.array([0.5, 0.32, levy(np.array([0.5, 0.32]))])])
