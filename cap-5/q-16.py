def inverse_mod(a,p):

    if a%p==0:
        return "p divide a"
    else:
        for i in range(1,p):
            if (i*a)%p==1:
                return i

# exemplo do teste 3
print (inverse_mod(7,47))
