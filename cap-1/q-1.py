'''fiz o algoritmo de euclides, mas não era suficiente
    foi preciso extender, material de consulta livro, wikipedia'''
def euclidean_algo(m,n):
    
    if m%n==0:
        return n

    else:
        remainder = m%n
        m = n
        n = remainder
        return str(euclidean_algo(m,n))

def mdc(m,n):

    mdc = euclidean_algo(m,n)
    result = "MDC: "+str(mdc)
    return result

# teste com número grande - funciona
#print (mdc(1221,1234567891011121314151617181920212223242526272829))

def extended_euclidean_algo(m, n):
    
    if m == 0:
        return (n, 0, 1)
    
    else:
        mdc, x, y = extended_euclidean_algo(n % m, m)
        alpha = y - ((n//m)*x)
        beta = x
        return (mdc, alpha, beta)

def pretty_eea(m,n):
    mdc, alpha,beta = extended_euclidean_algo(m,n)
    print ("Par de inteiros:  ",m,n)
    return ("mdc: " +str(mdc) + ", alpha: "+str(alpha)+", beta: "+ str(beta)+"\n")

print ("exemplo do livro: ",pretty_eea(1234,54))
print (pretty_eea(14,35))
print (pretty_eea(252,180))
print (pretty_eea(6643,2873))
print (pretty_eea(272828282,3242))

