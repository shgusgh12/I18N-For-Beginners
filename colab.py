import math
import numpy as np
import matplotlib.pyplot as plt
import sys

def falling():
    #exact graph
    def f1(t):
        result = 53.39*(1-np.exp((-0.18355)*t))
        return result
    
    h=2
    #approximate graph
    def f2(ti):
        tii = ti+ (9.8-12.5*ti/68.1)*h
        return tii
    #x1을 arrange 이용해 배열로 생성
    #x1 이용해 y1 배열 생성
    #plot 함수에 배열 넣어서 그래프 생성
    x1 = np.arange(0,1000,1)
    y1 = [f1(xx)for xx in x1]
    print('%5s, %10s'% ('t,s', 'v, m/s'))
    for i in range(7):
        print('%5d, %10.2f'% (x1[i*2], y1[i*2]))
    
    x2 = [0]
    y2 = [0]
    for i in range(1000):
        x2.append((i+1)*h) 
        y2.append(f2(y2[i]))

    

    fig, ax = plt.subplots()
    ax.plot(x1, y1)
    ax.plot(x2, y2)
    ax.set_xlim([0,13])
    plt.show()
#e = 1/2 일때 맥로린 함수
def mac():
    true_ans = np.exp(1/2)

    sol = 1
    ea = 100
    x = 1/2
    print('%5s %10.7s %10.7s %10.7s'%('Terms', 'Result', 'Et','Ea'))
    print('%5d %10d %10f'%(1, 1,(true_ans-1)/true_ans*100))
    for i in range(1,6):
        solold = sol
        sol = sol + x**i/math.factorial(i)
        if sol != 0:
            ea = abs((sol - solold)/sol)*100
        print('%5d %10.7f %10.7f %10.7f'%(i, sol, (true_ans-sol)/true_ans*100,ea))

def mac10():
    i = 1
    #i는 추가된 항의 개수
    sum = 1
    term = 1
    termold = 0
    print("i= 0 term= 1 sum = 1")
    while(1):
        if sum == termold:
            break
        termold = sum 
        term = 10**i/math.factorial(i)
        sum = termold + term
        print("i= ", i, "term= ", term,"sum= ", sum)
        i = i+1

#taylor series
#값을 이전 값에 누적하면서 계산 진행

def taylor():
    
    def cos(x,n):
        #그래프 그리려면 x의 모든 좌표 알아야한다 n은 항의 개수
        ans = 0
        for i in range(n):
            term = (-1)**i * x**(2*i)/math.factorial(2*i)
            ans += term
        return ans   
    #그래프 그리기 위해 x값 y값 설정
    #실제 코사인 그래프 
    angles  = np.arange(-2*np.pi, 2*np.pi, 0.1)
    cos_y = np.cos(angles)
    fig, ax = plt.subplots()


    #추정 코사인 그래프
    for i in range(1,6):
    #i - order approximation
        approx = [cos(angle, i) for angle in angles]
        ax.plot(angles, approx)
        print(i, cos(np.pi/3, i))


    
    ax.plot(angles, cos_y)
    ax.set_ylim([-5,5]) #그래프에서 보여지는 y범위 변경
    plt.show()

def taylorsin():
    
    def sin(x,n):
        #그래프 그리려면 x의 모든 좌표 알아야한다 n은 항의 개수
        ans = 0
        for i in range(n):
            term = (-1)**i * x**(2*i+1)/math.factorial(2*i+1)
            ans += term
        return ans   
    #그래프 그리기 위해 x값 y값 설정
    #실제 코사인 그래프 
    angles  = np.arange(-2*np.pi, 2*np.pi, 0.1)
    cos_y = np.sin(angles)
    fig, ax = plt.subplots()


    #추정 코사인 그래프
    for i in range(6):
    #i - order approximation
        approx = [sin(angle, i)for angle in angles]  #for문으로 리스트로 생성해야함
        ax.plot(angles, approx)
        print(i, sin(np.pi/3, i))

    ax.plot(angles, cos_y)
    ax.set_ylim([-7,5]) #그래프에서 보여지는 y범위 변경
    plt.show()
    
#x = 0.5 step size = 0.5 forward
def forward():
    h = 0.5
    
    def f(x):
        return -0.1 * x ** 4 - 0.15 * x ** 3 - 0.5 * x ** 2 - 0.25 * x + 1.2
    
    def ff(x):
        return (-0.4)*x**3+(-0.45)*x**2+(-1)*x+(-0.25) 
    tvalue = ff(0.5)
    def fff(x):
        xxx = (-1.2)*x**2+(-0.9)*x+(-1)

    # xrange = np.arange(0,1000,0.1)
    # yrange = [f(xranges)for xranges in xrange]
    # fig, ax = plt.subplots()
    # ax.plot(xrange, yrange)
    # plt.show()
    ans = (f(1) - f(0.5))/h
    t = (tvalue - ans)/tvalue *100
    center = (f(1) - f(0))/2*h
    t2 = (tvalue - center)/tvalue *100
    print (ans, t)
    print (center,t2)

#시작점 0.5 FF(0.5)인 값 추정
def center():
    h = 1
    x = 0.5
    
    def f(x): 
        return -0.1 * x ** 4 - 0.15 * x ** 3 - 0.5 * x ** 2 - 0.25 * x + 1.2
    def ff(x):
        return (-0.4)*x**3+(-0.45)*x**2+(-1)*x+(-0.25) 
    for i in range(11):
        ans = (f(x+h) - f(x-h))/(h*2)
        print('%20.10f %20.15f %20.15f' % (h, ans ,(ff(x)-ans)))
        h = h/10

#poly 

def poly():
    xi = 0
    h = 0.25 
    def f(x):
        return -0.1 * x ** 4 - 0.15 * x ** 3 - 0.5 * x ** 2 - 0.25 * x + 1.2
    def ff(x):
        return (-0.4)*x**3+(-0.45)*x**2+(-1)*x+(-0.25) 
    tvalue = ff(0.5)
    def fff(x):
        xxx = (-1.2)*x**2+(-0.9)*x+(-1)

    xrange = np.arange(0, 1000, 0.1)
    yrange = f(xrange)
    fig, ax = plt.subplots()
    ax.plot(xrange, yrange)
    

    zero = []
    first = []
    second = []
    for i in np.arange(0,1000,0.1):
        #zero.append(f(i))
        first.append(f(i)+ff(i)*h)
    ax.plot(xrange, first)
    ax.set_xlim([ 0,1])
    ax.set_ylim([ 0,1.2])
    plt.show()
    

def bisection():
  
  def f(x):
     return 9.8*68.1/x * (1-np.exp(-(x/68.1)*10)) - 40
  xl = 12
  xu = 16
  ea = 0
  for i in range (1,7):
      if i == 1:
        xr = 14
      xrold = xr
      xr = (xl + xu)/2
      print (i, xl , xu, xr)
      if f(xl) * f(xr) < 0:
        xu = xr
      elif f(xl) * f(xr) > 0:
        xl = xr
      ea = (f(xr) - f(xrold))/f(xr) * 100
      print(ea)


def newton():
    
    def f(x) : 
        return x**10 - 1
    def ff(x):
        return 10*x**9 
    x0 = 0.5
    xr = x0
    es = 0.0001
    imax = 10000
    iter = 0

    while True:
        xrold = xr
        xr = xrold - f(xrold)/ff(xrold)
        iter = iter + 1
        if xr != 0:
            ea = abs((xr - xrold))/xr * 100
        print(iter, xr, ea, es)
        if ea < es or iter >= imax:
            break

def secant():
    
    def f(c):
        return np.exp(-c)-c

    
    x0 = 0
    x1 = 1
    x2 = 

    

    

