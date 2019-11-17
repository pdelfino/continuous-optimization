# Algebra, Teoria dos Números e Criptografia 
---

 + Professor:  Luciano Castro.

 + Data: 2019.2.

 + Tech Stack: Python 3. 

 + Projeto: 
   
     - A1: 
     
        - Resolução das questões de programação do livro  "Números inteiros e Criptografia RSA" [S. C. Coutinho] do Capítulo 1 ao 6.
       
     - A2: 
     
        - [A definir, provável: blockchain]
        
     
### Por que usar github?

Conforme conversado um dia após a aula, apresentar o trabalho como arquivo markdown no github tem algumas vantagens:
- não exige nenhuma instalação, basta abrir o link no navegador;
- o formato markdown permite uma estética bacana que diferencia o texto de código do texto de linguahem natural;
- o trabalho fica no github e, como legado, faz parte do portfólio do aluno. Cada vez mais, o github tem sido analisado como parte da entrevista de programadores.



## Capítulo 1



### Questão 1

Na verdade, não era para fazer essa questão. A instrução é que apenas questões com indicação de "faça um programa..." sejam feitas.

No entanto, quando eu comecei o trabalho eu achei que era para fazer todas as questões que **pudessem ser resolvidas** por um algoritmo rs. Então, acabei fazendo a questão 1. Foi bacana porque servia como teste unitário para outras questões.



```python
'''fiz o algoritmo de euclides, mas não era suficiente
    foi preciso extender, material de consulta livro, wikipedia'''
def euclidean_algo(m,n):
    
    if m%n==0:
        return n

    else:
        remainder = m%n
        m = n
        n = remainder
        return str(euclidean_algo(m,n))

def mdc(m,n):

    mdc = euclidean_algo(m,n)
    result = "MDC: "+str(mdc)
    return result

# teste com número grande - funciona
#print (mdc(1221,1234567891011121314151617181920212223242526272829))

def extended_euclidean_algo(m, n):
    
    if m == 0:
        return (n, 0, 1)
    
    else:
        mdc, x, y = extended_euclidean_algo(n % m, m)
        alpha = y - ((n//m)*x)
        beta = x
        return (mdc, alpha, beta)

def pretty_eea(m,n):
    mdc, alpha,beta = extended_euclidean_algo(m,n)
    print ("Par de inteiros:  ",m,n)
    return ("mdc: " +str(mdc) + ", alpha: "+str(alpha)+", beta: "+ str(beta)+"\n")

print ("exemplo do livro: ",pretty_eea(1234,54))
print (pretty_eea(14,35))
print (pretty_eea(252,180))
print (pretty_eea(6643,2873))
print (pretty_eea(272828282,3242))


```



Output:

```python
Par de inteiros:   1234 54
exemplo do livro:  mdc: 2, alpha: -7, beta: 160

Par de inteiros:   14 35
mdc: 7, alpha: -2, beta: 1

Par de inteiros:   252 180
mdc: 36, alpha: -2, beta: 3

Par de inteiros:   6643 2873
mdc: 13, alpha: -16, beta: 37

Par de inteiros:   272828282 3242
mdc: 2, alpha: 697, beta: -58655556

```



### Questão 8

Essa questão já é exatamente o que foi pedido. Basicamente, trata-se de uma implementação do algoritmo estendido de Euclides, como indicado pelo enunciado. A solução é:

```python
'''
    fiz o algoritmo de euclides, mas não era suficiente
    foi preciso extender, material de consulta livro, wikipedia'''

def euclidean_algo(m,n):
    
    if m%n==0:
        return n

    else:
        remainder = m%n
        m = n
        n = remainder
        return euclidean_algo(m,n)

#print (euclidean_algo(5,12))

def mdc(m,n):

    mdc = euclidean_algo(m,n)
    result = "MDC: "+str(mdc)
    return result

#print (mdc(1221,1234567891011121314151617181920212223242526272829))

def extended_euclidean_algo(a, b,c):

    gcd = euclidean_algo(a,b)
     
    if gcd%c!=0:
        return "Sem solução"

    else:

        if a == 0:
            return (b, 0, 1)
        
        else:
            mdc, x, y = extended_euclidean_algo(b % a, a,c)
            alpha = y - ((b//a)*x)
            beta = x
            return (mdc, alpha, beta)

def pretty_eea(m,n,c):
    
    output = extended_euclidean_algo(m,n,c)
    if output=="Sem solução":
        return "Sem solução"
    else:

        mdc, alpha,beta = extended_euclidean_algo(m,n,c)
        print ("Par de inteiros:  ",m,n)
        return (str(alpha)+"*"+str(m)+" + "+str(beta)+"*" +str(n)+"="+str(c))

print ("Exemplo do livro: ",pretty_eea(1234,54,2))
print ("Exemplo similar ao do livro com a=1234, b=54 e c= 3: ",pretty_eea(1234,54,3))

```

Além do livro, consultei também o wikipedia sobre o assunto. Usei como teste um exemplo de dentro do livro, mas que não foi citado no enunciado:



```python
Par de inteiros:   1234 54
Exemplo do livro:  -7*1234 + 160*54=2
Exemplo similar ao do livro com a=1234, b=54 e c= 3:  Sem solução
```



### Questão 9

Achei essa questão bem legal. Na verdade, esse é o tipo de conteúdo que me empolga em matemática. Quando vi que o resultado do experimento computacional estava dando próximo ao resultado teórico tive uma sensação engraçada. Dá vontade de compartilhar com alguém rs.



```python
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
```



 A tabela gerada foi:

```python
Total de duplas de inteiros a serem avaliados: 10| Proporção co-primos: 0.3333333333333333

Total de duplas de inteiros a serem avaliados: 100| Proporção co-primos: 0.5656565656565656

Total de duplas de inteiros a serem avaliados: 1000| Proporção co-primos: 0.5885885885885885

Total de duplas de inteiros a serem avaliados: 10000| Proporção co-primos: 0.603060306030603

Total de duplas de inteiros a serem avaliados: 100000| Proporção co-primos: 0.6066560665606656

Total de duplas de inteiros a serem avaliados: 1000000| Proporção co-primos: 0.6083026083026083

Total de duplas de inteiros a serem avaliados: 10000000| Proporção co-primos: 0.6079405607940561
```

Resultado teórico: 6/(pi^2)

Aproximadamente, o resultado teórico é: 0.60792710185

O experimento computacional mais robusto, com 1000000 pares de número como entrada, retorna: 0.6079713607971361

Isto indica a convergência entre o resultado empírico e o resultado teórico



## Capítulo 2

### Questão 3

Novamente, eu fiz uma questão sem precisar. E, mais uma vez, acabou sendo um pouco produtivo.

No caso, trata-se do uso do algoritmo de Fermat para determinar os fatores de alguns números.

Meu código é:

```python
# algoritmo de fermat
# consulta ao livro basicamente
import math

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
print ("Fatore o número 17557: ",fermat_factor(17557))
print ("Fatore o número 455621: ",fermat_factor(455621))
print ("Fatore o número 731021: ",fermat_factor(731021))

#resultados foram checados em https://www.numberempire.com/numberfactorizer.php
```

O output é:

```python
Fatore o número 17557:  ('factor_1', 181.0, 'factor_2', 97.0)
Fatore o número 455621:  ('factor_1', 677.0, 'factor_2', 673.0)
Fatore o número 731021:  ('factor_1', 857.0, 'factor_2', 853.0)

```

Como grupo controle, eu usei o site https://www.numberempire.com/numberfactorizer.php para checar se o meu resultado estava correto.



### Questão 11

Essa questão de número altamente composto me causou um pouco de dúvida. Acho que a definição do Wikipedia me confundiu um pouco. Eu fiquei de tirar a dúvida com o Professor Luciano. Anotei na minha agenda duas vezes mas esqueci nas duas aulas.

Estranhamente, o wikipedia coloca o número 2 como um número altamente composto. Mas isso é estranho porque, afinal, o número 2 nem composto é! Veja: https://en.wikipedia.org/wiki/Highly_composite_number

Enfim, tirando essa confusão a parte, fiz o código:



```python
#função ingênua para contar quantos divisores tem um número
# obviamente, vou desconsiderar a divisão por 1 e pelo próprio número
# estranho, o wikipedia considera o 2 como HCN
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
```

O resultado retornado foi:

```python
[4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520]
```

Como propriedades básicas: todos são divisíveis por 2 e, com exceção do 4, todos são divisíveis por 3.

Outro fato curioso foi que, com a dúvida sobre o número 2, joguei o assunto no youtube. Acabei caindo no canal de um garoto de 9 anos chamado Simon Tiger. O garoto é bizarro. Veja a prova dele de que todos os fatoriais são HCN: https://www.youtube.com/watch?v=VyZhDWFx3Eo&t=527s. Ele já saiu na mídia algumas vezes e tem um conteúdo bem interessante. Um novo Gauss ou Terence Tao?

O código está rápido mas a função que fiz para contar os divisores foi bem ingênua. Como estava preocupado em fazer todas as questões, deixei para refatorar depois e acabei não conseguindo.



### Questão 12

Essa questão exigiu apenas uma adaptação do que eu já tinha feito. Usei, inclusive, o trabalho passado de base. O código que fiz foi:

```python
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
    
```

Ele retorna:

```python
Dois fatores de  226899561  são:  ('factor_1', 18989.0, 'factor_2', 11949.0)

```



## Capítulo 3

### Questão 10

Aqui foi necessário fazer o crivo de Eratóstenes. Essa é a parte do trabalho que mais me arrependo. Fiz um código que dá um retorno correto. Entretanto, está um pouco lento. Mas como estava preocupado em fazer todos antes de refatorar, passei para frente.  Não demora muito. Mas demora o suficiente para incomodar se a entrada for um inteiro expressivamente grande. O que está deixando lento é o "for" dentro do "while loop" que é desnecessário. A implementação foi direto da leitura do livro.



```python
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

```



O output é:

```python
Questão (1): f(x) = x^2 + 1:  [5, 17, 37, 101, 197, 257, 401, 577, 677, 1297, 1601, 2917, 3137, 4357, 5477, 7057, 8101, 8837] 

Questão (2): f(x) = x^2 -69x +1231:  [41, 43, 47, 53, 61, 71, 83, 97, 113, 131, 151, 173, 197, 223, 251, 281, 313, 347, 383, 421, 461, 503, 547, 593, 641, 691, 743, 797, 853, 911, 971, 1033, 1097, 1163, 1231, 1301, 1373, 1447, 1523, 1601, 1847, 1933, 2111, 2203, 2297, 2393, 2591, 2693, 2797, 2903, 3011, 3121, 3347, 3463, 3581, 3701, 3823, 3947, 4073, 4201] 

Questão (3): f(x) = 2x^2 -199:  [43, 89, 139, 193, 251, 313, 379, 449, 523, 601, 683, 769, 859, 953, 1051, 1153, 1259, 1483, 1601, 1723, 1979, 2113, 2251, 2393, 2539, 2689, 2843, 3001, 3163, 3329, 3499, 3673, 3851, 4219, 4409, 4603, 4801, 5003, 5209, 5419, 5851, 6073, 6299, 6529, 6763, 7001, 7243, 7489, 7993, 8513, 8779, 9049, 9323, 9601, 9883, 10169, 10459, 10753, 11353, 11969, 12601, 12923, 13249, 13913, 14251, 14593, 14939, 15289, 15643, 16001, 16363, 16729, 17099, 17851, 18233, 19009, 19403, 19801] 

Questão (4): f(x) = 8x^2 -530x + 7681:  [31, 79, 229, 281, 443, 499, 673, 733, 919, 983, 1181, 1249, 1459, 1531, 1753, 2063, 2143, 2389, 2473, 2731, 2819, 3089, 3181, 3463, 3559, 3853, 4259, 4363, 4789, 5119, 5231, 5573, 5689, 6043, 6163, 6529, 6653, 7159, 7549, 7681, 9199, 9781, 10993, 12269, 14303, 15013, 15739, 16481, 17239, 18013, 18803, 19609, 20431, 21269, 22123, 22993, 23879, 24781, 26633, 27583, 28549, 29531, 30529, 31543, 32573, 33619] 

```



### Questão 11

Achei questão bem interessante. Poderia ser um assunto de Análise Numérica. Fiquei curioso em quem inventou esse aproximador S(x). Muito curioso esse tanto de conta que parecem "arbitrárias" mas que acabam funcionando. O program para o problema é:

```python
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
```



O código retorna:

```python
| entrada:  11 | pi(x) - S(x):  7.7152975386241e-07 | pi(x) - [x/log x]: 0.4126436943332905
| entrada:  100 | pi(x) - S(x):  -0.0013903185453720823 | pi(x) - [x/log x]: 3.285275904837409
| entrada:  1000 | pi(x) - S(x):  0.10650770239291774 | pi(x) - [x/log x]: 23.235172698916045
| entrada:  2000 | pi(x) - S(x):  -1.1081836077476055 | pi(x) - [x/log x]: 39.873350152096236
| entrada:  3000 | pi(x) - S(x):  -1.3012546508164746 | pi(x) - [x/log x]: 55.298242348813744
| entrada:  4000 | pi(x) - S(x):  -3.1881138885775044 | pi(x) - [x/log x]: 67.72654209110874
| entrada:  5000 | pi(x) - S(x):  -2.410695899640359 | pi(x) - [x/log x]: 81.95214425345205
| entrada:  6000 | pi(x) - S(x):  -3.8748175416631057 | pi(x) - [x/log x]: 93.30635918693974
| entrada:  7000 | pi(x) - S(x):  -0.14699379367959864 | pi(x) - [x/log x]: 109.36700659224516
| entrada:  8000 | pi(x) - S(x):  -4.6117670404731825 | pi(x) - [x/log x]: 116.84479814577583
| entrada:  9000 | pi(x) - S(x):  -4.545671514177457 | pi(x) - [x/log x]: 128.52993833868913
| entrada:  10000 | pi(x) - S(x):  -1.156280659834465 | pi(x) - [x/log x]: 143.26379524187064
None
```



Como dá para ver:  **S(x) é uma aproximação bem melhor do que [x/log(x)].**



### Questão 12

Essa é outra questão interessante em que os experimentos computacionais corroboram o resultado teórico. O código é:

```python
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

    return ("| n: " + str(n) + "| pi_1_count: " + str(pi_1_count) + "| pi_3_count: " + str(pi_3_count) + "| proporção (pi_1/pi_3): " + str(prop))

print (pi_type(100000))
'''
iterar =100
while iterar<100000:
    
    print (pi_type(iterar))
    
    iterar+=100'''
```

Este código retorna:

```python
| n: 100000| pi_1_count: 4783| pi_3_count: 4808| proporção (pi_1/pi_3): 0.9948003327787022
```



Ou seja, para um n grande, a proporção é praticamente 1.



### Questão 13

Nas questões anteriores, meu erro de design do crivo de Eratóstenes não comprometeu a performance. Tudo funcionou rápido. Nesta questão, atrapalhou. Demorei a descobrir o valor: x=26861.

Mas está correto. Agora que sei o resultado, dá para rodar rápido. Demoro porque tive que testar para vários valores até achar um valor em que pi_1(x)>pi_3(x).

O código é:

```python
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

    return ("pi_1: ",pi_1_count,"pi_3: ", pi_3_count)

#print (pi_type(100000))

'''
iterar =3
while iterar<100000:
    
    p_1,p_3 = (pi_type(iterar))
    #print (p_1,p_3)
    #print (iterar)
    p_1+=0
    if p_1>p_3:
        print ("pi_1>pi_3 para n: ",iterar)
        break
    iterar+=2
'''
print ("caso em que pi_1>pi_3 x=26861: ",pi_type(26861))
```



E retorna **rapidamente**:

```python
caso em que pi_1>pi_3 x=26861:  ('pi_1: ', 1473, 'pi_3: ', 1472)

```

Lento foi achar esse valor de x... Para encontrá-lo usei a parte do código que está entre ''' de vermelho''' acima.



## Capítulo 4



### Questão 3

Novamente, fiz uma questão de cálculo de valores. E, mais uma vez, foi útil para fazer alguns testes e para ganhar maior confiança.

```python
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
```

Esse código retorna:

```python
5^20 (mod 7) equivale a: 4
7^1001 (mod 11) equivale a: 7
2^130 (mod 263) equivale a: 132
13^221 (mod 19) equivale a: 14

```



### Questão 11

 A solução dessa questão é bem curta:

```python
# recebe a,k e n -> fazendo a^k (mod n)

def potencia(base,expoente,mod):
    
    if mod==1:
        return 0
    
    result = 1

    for i in range(0,expoente):
        result = (result*base)%mod
    return (str(base)+"^"+str(expoente)+" (mod "+str(mod)+") equivale a: "+str(result))

#teste -> passou
print (potencia(4,13,497))
```

Implementação direta das instruções do livro que retorna:

```python
4^13 (mod 497) equivale a: 445

```



## Capítulo 5

### Questão 16

O enunciado fala para usar o exercício 14... Mas acaba nem sendo necessário. O código da inversa é curtinho. Importante garantir que "p" não divide "a".

```
def inverse_mod(a,p):

    if a%p==0:
        return "p divide a"
    else:
        for i in range(1,p):
            if (i*a)%p==1:
                return i

# exemplo do teste 3
print (inverse_mod(7,47))
```

Usei um site de cálculo online de inversa para checar se estava correto. O código acima retorna o resultado certo:

```python
27	
```



### Questão 17

Demorei um pouco a entender exatamente como chegaria na resposta. Depois de algumas contas no papel saiu:

```python

# preciso resolver [x^2 congruente a (mod p)]

def squared_modular(a,p):
    
    k = (p-3)/4
    
    sol_parcial = (a**(k+1))%p
    
    sol_final = []
    
    if (sol_parcial**2)%p==a%p:
    
        sol_final.append(sol_parcial)
        sol_final.append(-sol_parcial)
 
        return sol_final
    else:
        
        return ('A equação não possui resolução')

print ("Com x=0 temos (x^2) congruente a 1 (mod 1): ",squared_modular(1,1))
print ("Se  x^2 congruente a 2 (mod 7), então x=4 ou -4: ",squared_modular(2,7))
```

Que retorna:

```python
Com x=0 temos (x^2) congruente a 1 (mod 1):  [0.0, -0.0]
Se  x^2 congruente a 2 (mod 7), então x=4 ou -4:  [4.0, -4.0]

```



### Questão 18

Essa questão novamente envolve o Crivo e, mais uma vez, "carreguei"  o peso de uma implementação ininial sub-ótima. Nesse momento do trabalho eu comecei a pensar que era melhor ter corrigido lá atrás. Eu achava que os capítulos fossem ser mais independentes...

Mesmo subótimo, o código retorna o resultado desejado em menos de 30 segundos:

```python
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
```

Output:

```python
Fazendo os exemplos do livro. Com a=2,5,10,14 são 2 primos. E com a=19 são 5 primos, respectivamente: 
[1093, 3511]
[20771, 40487]
[3, 487]
[29, 353]
[3, 7, 13, 43, 137]
```



# Capítulo 6

### Questão 10

Essa questão ficou um pouco longa e exigiu uma função não ingênua para encontrar os fatores. Foi preciso alterar o algoritmo do Crivo, complementando-o.

```python
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

```

Ao aplicar o teste nos valores pedidos temos:

```python
base:  2 , teste:  2047
base:  3 , teste:  75
base:  5 , teste:  247
base:  7 , teste:  25
```



### Questão 11

Existem apenas dois exemplos de  base 2 em que r=5*10^4. Eles são: **1093, 3511**

Veja o código para chegar nesses valores abaixo. **Destaque para a parte final em que fiz testes de sanidade.**



```python
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

```

O retorno é:

```python
Os dois números são:  [1093, 3511]
Teste de Sanidade:  True

```

Curiosamente, o resultado não muda para r=5000 ou r=50000

Para r=50000 o código demora um pouco pela forma como implementei o algoritmo de eratóstenes



### Questão 9

Não consegui fazer essa questão. Falei com o colega Lucas Brito, que fez o curso ano passado. Ele me disse que também não tinha conseguido fazer uma das questões do livro. Imagino que seja essa.



### Questão 8

Também não consegui. Essa acho que faltou um pouco de tempo e de organização. Espero que as questões que fiz a mais, os comentários, esse arquivo na web, ou outro fator possa compensar de alguma forma.