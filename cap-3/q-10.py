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

#print (crivo_eratostenes(71))
def primo_polinomio(a,b,c):
    
    temp_lista = []
    
    for n in range(101):
        
        resultado = a*n**2+b*n+c
        
        if resultado not in temp_lista:
            temp_lista.append(resultado)
    
    primos = crivo_eratostenes(max(c,10000*a+100*b+c))
    
    lista_final= []
    
    for i in temp_lista:
        
        if i in primos:

            lista_final.append(i)
            lista_final.sort()
    
    return (lista_final)    

print ("Questão (1): f(x) = x^2 + 1: ",primo_polinomio(1,0,1),'\n')
print ("Questão (2): f(x) = x^2 -69x +1231: ",primo_polinomio(1,-69,1231),'\n')
print ("Questão (3): f(x) = 2x^2 -199: ",primo_polinomio(2,0,-199),'\n')
print ("Questão (4): f(x) = 8x^2 -530x + 7681: ",primo_polinomio(8,-530,7681),'\n')

