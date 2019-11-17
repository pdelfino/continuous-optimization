'''
    fiz o algoritmo de euclides, mas não era suficiente
    foi preciso extender, material de consulta livro, wikipedia'''

def euclidean_algo(m,n):
    
    if m%n==0:
        return n

    else:
        remainder = m%n
        m = n
        n = remainder
        return euclidean_algo(m,n)

#print (euclidean_algo(5,12))

def mdc(m,n):

    mdc = euclidean_algo(m,n)
    result = "MDC: "+str(mdc)
    return result

#print (mdc(1221,1234567891011121314151617181920212223242526272829))

def extended_euclidean_algo(a, b,c):

    gcd = euclidean_algo(a,b)
     
    if gcd%c!=0:
        return "Sem solução"

    else:

        if a == 0:
            return (b, 0, 1)
        
        else:
            mdc, x, y = extended_euclidean_algo(b % a, a,c)
            alpha = y - ((b//a)*x)
            beta = x
            return (mdc, alpha, beta)

def pretty_eea(m,n,c):
    
    output = extended_euclidean_algo(m,n,c)
    if output=="Sem solução":
        return "Sem solução"
    else:

        mdc, alpha,beta = extended_euclidean_algo(m,n,c)
        print ("Par de inteiros:  ",m,n)
        return (str(alpha)+"*"+str(m)+" + "+str(beta)+"*" +str(n)+"="+str(c))

print ("Exemplo do livro: ",pretty_eea(1234,54,2))
print ("Exemplo similar ao do livro com a=1234, b=54 e c= 3: ",pretty_eea(1234,54,3))

