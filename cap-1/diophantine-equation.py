def euclidean_algo(m,n):
    
    if m%n==0:
        #print ("m",m,"n",n)
        return n

    else:
        remainder = m%n
        m = n
        n = remainder
        #print ("m",m,"n",n)
        return euclidean_algo(m,n)

#print (euclidean_algo(10,6))
#print (euclidean_algo(21474836,21))
#print (euclidean_algo(1000000000,2010))

def diophantine(n,m):
    
    gcd = euclidean_algo(n,m)
    c = gcd

    return ((n/gcd)*(m/gcd))

print (diophantine(10,6))
