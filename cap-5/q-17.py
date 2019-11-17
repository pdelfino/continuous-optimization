
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
