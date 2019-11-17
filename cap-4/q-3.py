# recebe a,k e n -> fazendo a^k (mod n)

def potencia(base,expoente,mod):
    
    if mod==1:
        return 0
    
    result = 1

    for i in range(0,expoente):
        result = (result*base)%mod
    return (str(base)+"^"+str(expoente)+" (mod "+str(mod)+") equivale a: "+str(result))

#teste -> passo
# questões do livro resolvidas com o algoritmo da questão 11 - grupo controle: https://planetcalc.com/8326/
print (potencia(5,20,7))
print (potencia(7,1001,11))
print (potencia(2,130,263))
print (potencia(13,221,19))
