# recebe a,k e n -> fazendo a^k (mod n)

def potencia(base,expoente,mod):
    
    if mod==1:
        return 0
    
    result = 1

    for i in range(0,expoente):
        result = (result*base)%mod
    return (str(base)+"^"+str(expoente)+" (mod "+str(mod)+") equivale a: "+str(result))

#teste -> passou
print (potencia(4,13,497))
