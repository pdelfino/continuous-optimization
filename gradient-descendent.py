
x_initial = 3 # initial position for x is the point (3,0)

learning_rate = 0.01 # on the classical notation, the learning rate is the Mu parameter

tolerance = 0.000001 # tolerance definies when the results have finally converged

previous_step_size = 1 #

max_iterations = 10000 # parallelel to tolerance, max_iterations definies a maximum number of iterations

iterations_num = 0 # count the number of iterations

###
# usando o exemplo da derivada de f(x)= (x-3)^2
def gradient_function(x):
    return 2*(x-3)

while previous_step_size > tolerance or iterations_num < max_iterations:

    prev_x = x_initial #Store current x value in prev_x
    
    x_initial = x_initial - learning_rate * gradient_function(prev_x) #Grad descent
    
    previous_step_size = abs(x_initial - prev_x) #Change in x
    
    iterations_num = iterations_num+1 #iteration count
    
    print("Iteration",iterations_num,"\nX value is",x_initial) #Print iterations
    
print("The local minimum occurs at", x_initial)
