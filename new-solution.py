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


for i in range(1,101):
    print ("classic num. de passos:",(classic_grad(Q,b,x_initial,tolerance))[1],"| random num. de passos:", (stochastic_grad(Q,b,x_initial,tolerance))[1])
