# Continuous Optimization
---

 + Professor:  PhD Roger Behling.

 + Date: 2019.2.

 + Tech Stack: Python 3. 

     

### Problem statement

![](https://github.com/pdelfino/continuous-optimization/blob/master/enunciado.JPG)



### Solution

A solução passa pelo seguinte código:



```python
import numpy 
import random 

# example do wikipedia de matriz 3x3 (simétrica) definida positiva
Q = numpy.matrix([[2, -1,0], [-1,2, -1],[0,-1,2]])

b = numpy.matrix([[1],[1],[1]])

x_initial = numpy.matrix([[2],[2],[2]])

# equivale 10^(-10)
tolerance = 0.0000000001

def grad_f(x):
    
    return Q*x + b

def iteration_x_classic(x, n):

    alpha = -grad_f(x)
    
    divide_transpose = (numpy.dot(numpy.transpose(alpha),alpha)/numpy.dot(numpy.transpose(alpha),Q*alpha))

    return x + divide_transpose[0, 0] * alpha

def classic_grad(Q, b, x_initial, tolerance):
    
    x_final = x_initial
    
    x_next = iteration_x_classic(x_initial, 1)
    
    counter = 1
    
    aux_list = []
    
    while numpy.linalg.norm(x_next - x_final) > tolerance:
    
        aux_list.append(x_next)
        
        x_final = x_next
        
        x_next = iteration_x_classic(x_next, counter + 1)
        
        counter += 1
    
    return aux_list, counter

def iteration_x_stochastic(x, n):
    
    if (random.random() > 0.5): 
        sign_stochastic = 1
    
    else:
        sign_stochastic= -1
    
    random_value = (random.random() % 0.2) * sign_stochastic
    
    d_n = -grad_f(x) *(1 + random_value)
    
    return x + (numpy.dot(numpy.transpose(d_n), d_n) / numpy.dot(numpy.transpose(d_n), Q * d_n))[0, 0] * d_n

def stochastic_grad(Q, b, x_initial, tolerance):
    
    x_final = x_initial
    
    x_next = iteration_x_stochastic(x_initial, 1)
    
    counter = 1
    
    aux_list = []
    
    while numpy.linalg.norm(x_next - x_final) > tolerance:
        aux_list.append(x_next)
        x_final = x_next
        x_next = iteration_x_stochastic(x_next, counter + 1)
        counter += 1
    return aux_list, counter

print ("Número de passos do gradiente clássico:",(classic_grad(Q,b,x_initial,tolerance))[1])
print ("Número de passos do gradiente estocástico:",(stochastic_grad(Q,b,x_initial,tolerance))[1])

```



Inicialmente, de maneira ilustrativa, o código retorna:

```python
Número de passos do gradiente clássico: 60
Número de passos do gradiente estocástico: 40
```



Ambos os métodos possuem o mesmo objetivo: encontrar o valor mínimo da função. No exemplo ilustrativo acima, é possível perceber que o método do gradiente clássico levou mais tempo.

Entretanto, esse fato não é conclusivo, afinal, um dos métodos possui um componente aleatório e, consequentemente, uma certa variância. Assim, foi adicionado um pequeno trecho de código que gera uma simulação repetindo o mesmo processo 100 vezes:



```python
for i in range(1,101):
    print ("classici num. de passos:",(classic_grad(Q,b,x_initial,tolerance))[1],"| random num. de passos:", (stochastic_grad(Q,b,x_initial,tolerance))[1])
```



O resultado foi a seguinte tabela:

```python

classic num. de passos: 60 | random num. de passos: 43
classic num. de passos: 60 | random num. de passos: 34
classic num. de passos: 60 | random num. de passos: 45
classic num. de passos: 60 | random num. de passos: 46
classic num. de passos: 60 | random num. de passos: 36
classic num. de passos: 60 | random num. de passos: 54
classic num. de passos: 60 | random num. de passos: 43
classic num. de passos: 60 | random num. de passos: 38
classic num. de passos: 60 | random num. de passos: 48
classic num. de passos: 60 | random num. de passos: 54
classic num. de passos: 60 | random num. de passos: 42
classic num. de passos: 60 | random num. de passos: 38
classic num. de passos: 60 | random num. de passos: 53
classic num. de passos: 60 | random num. de passos: 33
classic num. de passos: 60 | random num. de passos: 46
classic num. de passos: 60 | random num. de passos: 38
classic num. de passos: 60 | random num. de passos: 48
classic num. de passos: 60 | random num. de passos: 46
classic num. de passos: 60 | random num. de passos: 39
classic num. de passos: 60 | random num. de passos: 30
classic num. de passos: 60 | random num. de passos: 47
classic num. de passos: 60 | random num. de passos: 53
classic num. de passos: 60 | random num. de passos: 55
classic num. de passos: 60 | random num. de passos: 50
classic num. de passos: 60 | random num. de passos: 36
classic num. de passos: 60 | random num. de passos: 57
classic num. de passos: 60 | random num. de passos: 49
classic num. de passos: 60 | random num. de passos: 47
classic num. de passos: 60 | random num. de passos: 47
classic num. de passos: 60 | random num. de passos: 52
classic num. de passos: 60 | random num. de passos: 52
classic num. de passos: 60 | random num. de passos: 46
classic num. de passos: 60 | random num. de passos: 42
classic num. de passos: 60 | random num. de passos: 37
classic num. de passos: 60 | random num. de passos: 57
classic num. de passos: 60 | random num. de passos: 44
classic num. de passos: 60 | random num. de passos: 55
classic num. de passos: 60 | random num. de passos: 40
classic num. de passos: 60 | random num. de passos: 54
classic num. de passos: 60 | random num. de passos: 54
classic num. de passos: 60 | random num. de passos: 47
classic num. de passos: 60 | random num. de passos: 54
classic num. de passos: 60 | random num. de passos: 54
classic num. de passos: 60 | random num. de passos: 55
classic num. de passos: 60 | random num. de passos: 43
classic num. de passos: 60 | random num. de passos: 44
classic num. de passos: 60 | random num. de passos: 30
classic num. de passos: 60 | random num. de passos: 33
classic num. de passos: 60 | random num. de passos: 41
classic num. de passos: 60 | random num. de passos: 40
classic num. de passos: 60 | random num. de passos: 48
classic num. de passos: 60 | random num. de passos: 48
classic num. de passos: 60 | random num. de passos: 40
classic num. de passos: 60 | random num. de passos: 40
classic num. de passos: 60 | random num. de passos: 35
classic num. de passos: 60 | random num. de passos: 49
classic num. de passos: 60 | random num. de passos: 44
classic num. de passos: 60 | random num. de passos: 45
classic num. de passos: 60 | random num. de passos: 47
classic num. de passos: 60 | random num. de passos: 47
classic num. de passos: 60 | random num. de passos: 49
classic num. de passos: 60 | random num. de passos: 41
classic num. de passos: 60 | random num. de passos: 47
classic num. de passos: 60 | random num. de passos: 55
classic num. de passos: 60 | random num. de passos: 42
classic num. de passos: 60 | random num. de passos: 48
classic num. de passos: 60 | random num. de passos: 50
classic num. de passos: 60 | random num. de passos: 51
classic num. de passos: 60 | random num. de passos: 41
classic num. de passos: 60 | random num. de passos: 40
classic num. de passos: 60 | random num. de passos: 45
classic num. de passos: 60 | random num. de passos: 42
classic num. de passos: 60 | random num. de passos: 40
classic num. de passos: 60 | random num. de passos: 36
classic num. de passos: 60 | random num. de passos: 54
classic num. de passos: 60 | random num. de passos: 40
classic num. de passos: 60 | random num. de passos: 49
classic num. de passos: 60 | random num. de passos: 57
classic num. de passos: 60 | random num. de passos: 45
classic num. de passos: 60 | random num. de passos: 54
classic num. de passos: 60 | random num. de passos: 37
classic num. de passos: 60 | random num. de passos: 52
classic num. de passos: 60 | random num. de passos: 48
classic num. de passos: 60 | random num. de passos: 40
classic num. de passos: 60 | random num. de passos: 43
classic num. de passos: 60 | random num. de passos: 47
classic num. de passos: 60 | random num. de passos: 57
classic num. de passos: 60 | random num. de passos: 41
classic num. de passos: 60 | random num. de passos: 51
classic num. de passos: 60 | random num. de passos: 52
classic num. de passos: 60 | random num. de passos: 46
classic num. de passos: 60 | random num. de passos: 39
classic num. de passos: 60 | random num. de passos: 45
classic num. de passos: 60 | random num. de passos: 35
classic num. de passos: 60 | random num. de passos: 42
classic num. de passos: 60 | random num. de passos: 51
classic num. de passos: 60 | random num. de passos: 44
classic num. de passos: 60 | random num. de passos: 41
classic num. de passos: 60 | random num. de passos: 45
classic num. de passos: 60 | random num. de passos: 44
```



Como era de se esperar, o método clássico é **determinístico** e, pela sua natureza, retornou sempre o mesmo número de passos: 60.

O método estocástico, por sua vez, retornou, **na absoluta maioria das vezes**, um número de passos inferior ao método clássico. **Sendo, portanto, mais rápido.**



