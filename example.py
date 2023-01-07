t = int(input())

zero = 0
one = 0
def fibo(n):
    if n == 0:
        global zero
        zero = zero + 1
        return 0
    elif n == 1:
        global one
        one = one + 1
        return 1
    else:
        return fibo(n-1) + fibo(n-2)

for i in range(t):
    cnt_0 = [1, 0] #fibo 0
    cnt_1 = [0, 1] #fibo 1
    fibo(2)  = fibo(0) + fibo(1)  [1,1]
    fibo(3) = fibo(2) + fibo(1)

    x = int(input())
   