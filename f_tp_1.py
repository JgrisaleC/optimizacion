import numpy as np

# Puntos iniciales de cada funcion (estan en orden)
puntos_iniciales = [np.array([-2, 1]), np.array([-0.21691, -0.122605]), np.array([2, 1]), np.array([-1.2, 1.]),
                    np.array([-2, 2]), np.array([0.4, 1., 0.]), np.array([0, 10, 20]), np.array([1, 1]),
                    np.array([0.5, 0.5]), np.array([-2, 1]), np.array([0.5, 0.5, 0.5, 0.5]), np.array([-3, -1, -3, -1]),
                    np.array([0.3, 0.4]), np.array([-1, -1, -1]), np.array([1, 1]), np.array([-1, 2]),
                    np.array([-8, -5]), np.array([3, 2]), np.array([1, 2]), np.array([2, -1]), np.array([2, 0])]


# x0 = np.array([-2, 1])
def f0(x, d=2):
    return -0.1 * sum(np.cos(5 * np.pi * x[i]) for i in range(d)) + sum(x[i] ** 2 for i in range(d))


# x0 = np.array([-0.21691, -0.122605])
def f1(x):
    s1 = np.exp(-x[0] ** 2 - x[1] ** 2)
    s2 = np.exp(-(x[0] - 1) ** 2 - (x[1] - 1) ** 2)
    return (s1 - s2) * 2


# x0 = np.array([2, 1])
def f2(x):
    s1 = np.exp(-x[0] ** 2 - x[1] ** 2)
    s2 = np.exp(-(x[0] - 1) ** 2 - (x[1] - 1) ** 2)
    return (s1 - s2) * 2


# x0 = np.array([-1.2, 1])
def f3(x):
    return sum(100*(x[i+1]-x[i]**2)**2+(x[i]-1)**2 for i in range(2-1))


# x0 = np.array([-2, 2])
def f4(x):
    return (x[0]-2)**2 + (x[1]+1)**2


# x0 = np.array([0.4, 1, 0])
def f5(x):
    c = [0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420, 0.3521, 0.3989]
    y = np.array(c + list(reversed(c[:-1])))
    return sum((x[0]*np.exp((-x[1]*(((8-i)/2 - x[2])**2))/2)-y[i])**2 for i in range(15))


# x0 = np.array([0, 10, 20])
def f6(x):
    return sum((np.exp(-0.1*i*x[0])-np.exp(-0.1*i*x[1])-x[2]*(np.exp(-0.01*i)-np.exp(-10*0.1*i)))**2 for i in range(4))


# x0 = np.array([1,1])
def f7(x):
    return (1.5-x[0] + x[0]*x[1])**2 + (2.25-x[0] + x[0]*x[1]**2) ** 2 + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2


# x0 = np.array([0.5, 0.5])
def f8(x, d=2):
    return sum((d - sum(np.cos(x[j]) for j in range(d)) + i*(1-np.cos(x[i])) - np.sin(x[i]))**2 for i in range(d))


# x0 = np.array([-2, 1])
def f9(x):
    return np.cos(x[0]) ** 2 + np.sin(x[1]) ** 2


# x0 = np.array([0.5, 0.5, 0.5, 0.5])
def f10(x, d=4):
    return sum((x[i] + sum(x[j] for j in range(d)) - (d + 1))**2 for i in range(d)) + (np.prod(x) - 1)**2


# x0 = np.array([-3, -1, -3, -1])
def f11(x):
    s1 = 10*(x[1] - x[0]**2)
    s2 = 1 - x[0]
    s3 = np.sqrt(90) * (x[3] - x[2]**2)
    s4 = 1 - x[2]
    s5 = np.sqrt(10) * (x[1] + x[3] - 2)
    s6 = np.sqrt(10) * (x[1] - x[3])
    return s1**2 + s2**2 + s3**2 + s4**4 + s5**2 + s6**2


# x0 = np.array([0.3, 0.4])
def f12(x):
    s1 = 2 + 2 - (np.exp(x[0]) + np.exp(x[1]))
    s2 = 2 + 4 - (np.exp(2*x[0]) + np.exp(2*x[1]))
    return s1**2 + s2**2


# x0 = np.array([-1, -1, -1])
def f13(x):
    s1 = (3-2*x[0])*x[0] - 2*x[1] + 1
    s2 = (3-2*x[1])*x[1] - x[0] - 2*x[2] + 1
    s3 = (3-2*x[2])*x[2] - x[1] + 1
    return s1**2 + s2**2 + s3**2



# x0 = np.array([1, 1])
def f14(x):
    return x[0] ** 2 + 2 * (x[1] ** 2) - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7


# x0 = np.array([-1, 2])
def f15(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


# x0 = np.array([ -8, -5])
def f16(x):
    return (x[0] + 10)**2 + (x[1] + 10)**2 + np.exp(-x[0]**2 - x[1]**2)


# x0 = np.array([3, 2])
def f17(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2)


# x0 = np.array([1, 2])
def f18(x):
    return 0.6 + sum((np.sin(1 - 16 / 15 * x[i]))**2 - 1 / 50 * np.sin(4 - 64 / 15 * x[i]) -
                     np.sin(1 - 16 / 15 * x[i]) for i in range(2))


# x0 = np.array([2, -1])
def f19(x):
    return x[0]**4 + x[1]**4 + 2*(x[0]**2)*(x[1]**2) - 4*x[0] + 3


# x0 = np.array([2, 0])
def f20(x):
    return 100 * (x[1] - x[0] ** 3) ** 2 + (1 - x[0]) ** 2
