import random

def euclidean_algo(m,n):
    
    if m%n==0:
        return n

    else:
        remainder = m%n
        m = n
        n = remainder
        return euclidean_algo(m,n)

#função para gerar pares de números
def random_pairs(n):
    lista = []
    for i in range(1,n):
        pairs = []

        pairs.append(random.randint(1,10000000))
        pairs.append(random.randint(1,10000000))
        lista.append(pairs)
    return lista

def mdc_on_list(l):

    num_pares = len(l)
    counter_coprime = 0

    for i in l: 
        #print (i)
        mdc = euclidean_algo(i[0],i[1])
        #print (mdc) 
        if mdc==1:
            counter_coprime+=1
            #print (counter_coprime)

    quotient = counter_coprime/num_pares
    #print (quotient)

    return quotient


for i in [10,100,1000,10000,100000,1000000,10000000]:

    print ("Total de duplas de inteiros a serem avaliados: "+str(i)+"| Proporção co-primos: "+str(mdc_on_list(random_pairs(i))))
