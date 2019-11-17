#função ingênua para contar quantos divisores tem um número
# obviamente, vou desconsiderar a divisão por 1 e pelo próprio número
# estranho, o wikipedia considera o 2
# video do youtube do simon tiger
# https://www.youtube.com/watch?v=VyZhDWFx3Eo&t=478s

def count_div(n):
    
    count = 0
    
    for i in range(2,n):
        
        if n%i==0:
            
            count+=1
    
    return count

#print (count_div(30))

def num_all_divisors(n):
    
    div_num_list = []

    for i in range(1,n+1):
        div_num_list.append([i,count_div(i)])

    return div_num_list

#print (num_all_divisors(10))

def list_all_hcn(n):

    list_num_and_div =  num_all_divisors(n)
    
    maximo = 0
    
    lista_hcn = []
    
    for i in list_num_and_div:
        
        if maximo<i[1]:
            maximo=i[1]
            lista_hcn.append(i[0])

    return lista_hcn

print (list_all_hcn(5000))
