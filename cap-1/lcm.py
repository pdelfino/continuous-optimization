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


def lcm(a,b):
    
    gcd = euclidean_algo(a,b)
    return int((a*b)/gcd)

print (lcm(3,5))
print (lcm(790933790548 ,7))
print (lcm(790933790547,1849639579327))
