import numpy as np

#base passada pelo enunciado
base_enunciado = [2,3,5,7]

def count_factors(numero):
    
    n = numero
    fator = []
    contador = 0
    
    for a in range(2,int(numero/2)+1):
    
        while numero%a == 0:
        
            numero/=a     
            contador+=1
        
        if contador!=0:
            fator.append(contador)
        
        contador=0
    
    if n==numero:
    
        fator.append(1)
    
    numero_fatores=1
    
    for expoente in fator:
        numero_fatores*=expoente+1
    
    return numero_fatores


def crivo_eratostenes_compl(n):
    
    if n%2 == 0:
        crivo_eratostenes_compl(n-1)
    
    v = np.ones((int((n-1)/2)))
    
    P = 3
    
    while P**2<=n:
    
        if v[int((P-1)/2)-1]==0:
            P+=2
        
        else:
            T=P**2
        
            while T<n:
                v[int((T-1)/2)-1]=0
                T+=2*P
            P+=2
    l=[]
    
    for m in range(len(v)):
        
        if v[m]==0:
            l.append(2*(m+1)+1)
    
    if count_factors(l[-1])==2:
        l.remove(l[-1])
    
    return l

def miller_test(n,b):
    
    k = 0
    
    N = n-1
    
    while (N)%2 == 0:
        
        N = N/2
        
        k = k+1
    
    q = (n-1)/(2**k)
    
    i = 0
    
    r = (b**q)%n
    
    while True:
        
        if i==0 and r==1:
            return "teste não conclusivo"
        
        elif i>=0 and r==n-1:
            return "teste não conclusivo"
        
        i = i + 1
        
        r=(r**2)%n
        
        if i>=k:
            return('composto')

def pseudo_forte(b):
    
    lista_pseudoprimos = crivo_eratostenes_compl(10000)
    
    for i in lista_pseudoprimos:
         
        if miller_test(i,b) == "teste não conclusivo":
            
            return i

for i in base_enunciado:
    
    print ("base: ",i,", teste: ",pseudo_forte(i))

