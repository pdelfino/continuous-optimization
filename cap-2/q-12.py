# algoritmo de fermat
import math
import random

def fermat_core(n):

    x = (n)**(1/2)
    x = int(x)
    x += 1
    #print (x)

    if x**2==n:
        #print ("primeiro if")
        return "n primo"
    
    else:
        #print ("x",x,"n", n)
        y= math.sqrt((x**2)-n)
        #print ("x",x,"y",y)
        
        stop = (n+1)/2 
        
        while (y%1!=0) or(x==stop):
            x += 1
            y= math.sqrt((x**2)-n)

        return (x,y)

def fermat_factor(n):

    x,y = fermat_core(n)
    factor_1,factor_2 = (x+y),(x-y)
    return ("factor_1",factor_1,"factor_2",factor_2)

#print (fermat_core(1342127))
#print (fermat_factor(1342127))
#print ("Fatore o número 17557: ",fermat_factor(17557))
#print ("Fatore o número 455621: ",fermat_factor(455621))
#print ("Fatore o número 731021: ",fermat_factor(731021))

#sortear um número grande maior que 1 bi (10**9) e menor que 2**32
random_num = random.sample(range(10**7, 2**32), 1)
print (random_num)

for i in random_num:
    print ("Dois fatores de ",i," são: ",fermat_factor(i)) 
    
