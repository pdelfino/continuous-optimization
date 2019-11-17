import math

a_i = [229168.50747390,-429449.7206839,199330.41355048,28226.22049280,0,0,-34712.81875914,0,
       33820.10886195,-25379.82656589,8386.14942934,-1360.44512548,89.14545378]

#decompus S várias vezes
def series(x):
    
    coeficients = a_i
    total_sum = 0

    for i in range(0,len(coeficients)):
        first_log = math.log(x)
        second_log = math.log(first_log)
        final_log = second_log**(i)
        partial_sum = coeficients[i]*(final_log)
        total_sum += partial_sum
        #print ("i",i,"first_log",first_log,"second_log",second_log,"final_log",final_log,
        #"partial_sum",partial_sum)
        #print (i,total_sum)
    return total_sum

def power_quarter(x):

    return (x)**(-1/4)

def log_quot(x):
    
    return x/(math.log(x))

def S(x):

    return (log_quot(x))*(1+ (power_quarter(series(x))))

#ok, valores razoáveis para S
#print (S(1000))

def pi(x):
    if x==11:
       return 5

    elif x==100:
        return 25
    
    elif x==1000:
        return 168
    
    elif x==2000:
        return 303
    
    elif x==3000:
        return 430 

    elif x==4000:
        return 550

    elif x==5000:
        return 669 
    
    elif x==6000:
        return 783

    elif x==7000:
        return 900            
    
    elif x==8000:
        return 1007

    elif x==9000:
        return 1117

    elif x==10000:
        return 1229 
    else:
        return None

input_ex = [11,100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]

def tabela_comparative(entrada):

    for i in entrada:
        print ("| entrada: ",i,"| pi(x) - S(x): ",pi(i)-S(i),"| pi(x) - [x/log x]:",pi(i)-(i/math.log(i)))
    return None
print (tabela_comparative(input_ex))


#comentário final: S(x) é uma aproximação bem melhor do que [x/log(x)]
