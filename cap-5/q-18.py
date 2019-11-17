import math

def crivo_eratostenes(n):

    stop =math.ceil(math.sqrt(n))
    
    lista = list(range(3,n+1))
    lista = lista[::2]
    
    iter_index_crivo = 0
    iter_crivo = lista[iter_index_crivo]
    
    while iter_crivo<stop:

        for i in lista[iter_index_crivo::]:

            #print("i",i,"iter_crivo",iter_crivo)
            if i%iter_crivo==0 and (iter_crivo!=i):
                #print ("lista antes da alteração: ",lista)
                lista.remove(i)
                #print ("lista depois da alteração: ",lista)
        
        iter_index_crivo += 1
        iter_crivo = lista[iter_index_crivo]
       
    return lista

#print (crivo_eratostenes(100))
# função para checar se é da forma 4n+1 ou 4n+3
# n é o alcance

def primos_congruentes(a,r):
    
    result = []
    
    num_primos = crivo_eratostenes(r)
    
    for p in num_primos:
    
        if  (a**(p-1))%(p**2)==1:
            result.append(p)
    
    return result  

print ("Fazendo os exemplos do livro. Com a=2,5,10,14 são 2 primos. E com a=19 são 5 primos, respectivamente: ")
print (primos_congruentes(2,100000))
print (primos_congruentes(5,100000))
print (primos_congruentes(10,100000))
print (primos_congruentes(14,100000))
print (primos_congruentes(19,100000))
