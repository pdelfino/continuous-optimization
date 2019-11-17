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

def pi_type(n):

    lista_primos = crivo_eratostenes(n)
    #print(lista_primos)
    pi_1 = []
    
    pi_3 = []

    for i in lista_primos:
        
        equation = (i -1)%4
        if equation==0:
            pi_1.append(i)
        else:
            pi_3.append(i)

    pi_1_count = len(pi_1)
    pi_3_count = len(pi_3)
    prop = pi_1_count/pi_3_count

    return (pi_1_count, pi_3_count)

#print (pi_type(100000))

iterar =3
while iterar<100000:
    
    p_1,p_3 = (pi_type(iterar))
    #print (p_1,p_3)
    #print (iterar)
    p_1+=0
    if p_1>p_3:
        print ("pi_1>pi_3: ",p_1)
        break
    iterar+=2
