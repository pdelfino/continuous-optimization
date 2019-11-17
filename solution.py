import numpy

n = 2 # the matrix is 2x2

Q= [[1, 0],[0,1]] # identity matrix, which is a squared definite positive matrix

b = [[2], [2]] # by the problem statement, this is an arbitrary value

def problem_function(x):
    
    x_transpose = numpy.transpose(x)

    partial_first = numpy.matmul(x_transpose, Q)
    #print (partial_first)
    partial_first = numpy.matmul(partial_first,x)
    #print (partial_first)
    partial_first = (1/2)*partial_first
    partial_second = numpy.matmul(x_transpose,b)
    #print (partial_second)
    total = partial_first + partial_second

    return total

for i in range(1,10):
    print (problem_function([i, i]))


