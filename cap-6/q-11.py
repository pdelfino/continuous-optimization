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


def pseudo_primo_quadrado(r):
    
    num_primos = crivo_eratostenes(r)
    lista_temp = []

    for primo in num_primos:
        
        resto = 1
        mod_externo = primo**2
        fator_primeiro = (2**(primo-1))
        fator_segundo =  ((2**(primo))%(primo**2))

        if ((fator_primeiro*((fator_segundo)**(primo-1))))%(mod_externo)==resto:
            
            lista_temp.append(primo)
    
    lista_final = lista_temp

    return lista_final

print("Os dois números são: ",pseudo_primo_quadrado(5000))

# fazer teste de sanidade
def teste_sanidade(par):
    
    output = pseudo_primo_quadrado(par)

    for i in output:

        hold = 2**(i**2-1)
        if (hold%i**2)!=1:
            return False

    return True

print ("Teste de Sanidade: ",teste_sanidade(5000))
