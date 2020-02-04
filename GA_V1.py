"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Created on Sun Dec  1 18:56:23 2019
@author: mpervej
This Code does: Maximize Peak Function using Cannonical GA
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d
import timeit

start = timeit.default_timer()
plt.close('all')
title_font = {'fontname':'Times New Roman', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} 
axis_font = {'fontname':'Times New Roman', 'size':'14'}
plt.rcParams["font.family"] = "Times New Roman"

#========================================= Problem 2 #=========================================
def objfunc(x1,x2):                                                                                                                # Fitness Function for given problem
    f = 3*(1-x1)**2*np.exp(-x1**2 - (x2+1)**2) - 10*(x1/5 - x1**3 - x2**5)*np.exp(-x1**2-x2**2) - np.exp(-(x1+1)**2 - x2**2)/3      # Peak Function
    return f
#======================= Surface Plot ==========================
a0, b0 = -3, 3
#xlist = np.linspace(a0, b0, 50)
#xx, yy = np.meshgrid(xlist, xlist)
#fa = objfunc(xx, yy)
#no_of_contours = 20
#
#fig = plt.figure()
#ax = fig.gca(projection = '3d')
#surf = ax.plot_surface(xx, yy, objfunc(xx,yy),linewidth = 1, color = 'g', antialiased = True)
#ax.set_title('Surface of the $Peak$ function', **title_font)
#ax.set_xlabel('$x_1$', **axis_font)
#ax.set_ylabel('$x_2$', **axis_font)
#ax.set_zlabel('$f(x_1,x_2)$', **axis_font)
#plt.autoscale(enable = True, axis = 'all', tight = True)
#plt.show()
##plt.savefig("Results/surf_peak.eps", dpi=1200, format='eps', orientation='portrait')
#
##======================== Part (b) ===========================
#fig, ax = plt.subplots()
#ax.scatter(xx, yy, marker = '.', Linewidth = '0.1', color = 'y')
#da = ax.contour(xlist, xlist, fa, no_of_contours, extend3d = True)
#ax.clabel(da, fontsize = 10, inline = 1)
#ax.set_xlabel('$x_1$')
#ax.set_ylabel('$x_2$')
#ax.set_title(r'Contours of the $Peak$ function', **title_font)
##plt.savefig("Results/cntr_peak.eps", dpi=1200, format='eps', orientation='portrait')


def decoding(min_val, max_val, encod_level, binary_string):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Self-defined binary decoding: Return Decimal value
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if len(binary_string) != encod_level:
        print('Error in Decoding: len of binary string is not equal to encoding level')
    temp = np.zeros((encod_level, 1))
    for i in range(encod_level):
        temp[i,0] = binary_string[i]*2**(encod_level - 1 - i)    
    deci_decode = min_val + ((max_val-min_val)/(2**encod_level - 1))*np.sum(temp)
    return deci_decode

def fitness_evaluation(pop, min_val, max_val, encod_level):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Evaluate the fitness function; Return the function value for all Chromosomes
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    fun_val = np.zeros((total_pop_size,1))
    for i in range(len(pop)):
        x1 = decoding(min_val, max_val, int(encod_level/2), list(pop[i, 0:int(encod_level/2)]))
        x2 = decoding(min_val, max_val, int(encod_level/2), list(pop[i, int(encod_level/2):encod_level]))
        fun_val[i,0] = objfunc(x1, x2)
    return fun_val
        
def Selection(pop, min_val, max_val, encod_level):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Return "a choromose" based on Roulette Wheel
    Same size as the size of a chromose: (Dimension x Ecoding Level) (2 X 5 in our case)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Int_Evaluation = fitness_evaluation(pop, min_val, max_val, encod_level)
    Selection_Probability = (Int_Evaluation/np.sum(Int_Evaluation))
    Selection_Probability = Selection_Probability/np.sum(Selection_Probability)
#    Ind = np.where(np.random.multinomial(1, Selection_Probability))[0][0]          # Using Multinomial Distribution
    g_cum = np.cumsum(Selection_Probability)
    Ind = np.where(g_cum - np.random.rand() > 0)[0][0]                              # Using CumSum 
    Selected_gen = np.array([pop[Ind,:]])
    return Selected_gen

def Mating(pop, min_val, max_val, encod_level):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Return Mating Pool Using the Selection based on Roulette Wheel
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Mating_Pool = np.zeros(shape = (total_pop_size, encod_level))
    for j in range(total_pop_size):
        Mating_Pool[j,:] = Selection(pop, min_val, max_val, encod_level)
    return Mating_Pool

def Crossover(pop, min_val, max_val, encod_level):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Perform Crossover and Return New Population:
        Choose: (1) two chormosomes and (2) crossover point, randomly
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    new_pop = np.zeros(np.shape(pop))
    ii = 0
    while ii <= total_pop_size-2:
        pop_temp0 = pop[np.random.randint(0, total_pop_size), :]
        pop_temp1 = pop[np.random.randint(0, total_pop_size), :]
        crossover_point = np.random.randint(1, encod_level-1)
        crossover0 = np.hstack( (pop_temp0[0:crossover_point], pop_temp1[crossover_point:len(pop_temp0)]))
        crossover0 = np.reshape(crossover0, (1, encod_level)) 
        crossover1 = np.hstack( (pop_temp1[0:crossover_point], pop_temp0[crossover_point:len(pop_temp0)]))
        crossover1 = np.reshape(crossover1, (1, encod_level))     
        new_pop[ii, :] = crossover0
        new_pop[ii+1 :] = crossover1
        ii += 2
    return new_pop

def Mutation(pop, encod_level, mutation_rate):
    new_pop = np.zeros(np.shape(pop))
    for i in range(total_pop_size):
        for ii in range(encod_level):
            if np.random.random() < mutation_rate:
                if pop[i,ii] == 0:
                    new_pop[i,ii] = 1
                elif pop[i,ii] == 1:
                    new_pop[i,ii] = 0
            else:
                new_pop[i,ii] = pop[i,ii]
    return new_pop

# Initialization
total_pop_size = 100
encoding_level = 32
pop_dimension = 2
max_iter = 200
generation_best = []
generation_worst = []
generation_avg = []
population = np.random.randint(0, 2, size = (total_pop_size, encoding_level))
ini_val = fitness_evaluation(population, a0, b0, encoding_level)

generation_best.append(max(ini_val)[0])
generation_worst.append(min(ini_val)[0])
generation_avg.append(sum(ini_val)/total_pop_size)
gbest = ini_val[np.argmax(ini_val, axis=0)[0]][0]
print('Iteration No : ', 1,';', 'Generation Best Value: ', max(ini_val)[0])

# Main Loop
cnt = 1
while cnt <= max_iter-1:     
    new_pop = np.zeros(np.shape(population))
    # Mating Pool
    Mating_Pool = Mating(population, a0, b0, encoding_level) 
    
    # Crossover
    new_pop = Crossover(Mating_Pool, a0, b0, encoding_level)
    
    gen_evaluation = fitness_evaluation(new_pop, a0, b0, encoding_level)
    gen_best = max(gen_evaluation)[0]
    if gen_best > gbest:
        gbest = gen_best 
        temp = new_pop[np.argmax(gen_evaluation, axis=0)[0],:]
        x1 = decoding(a0, b0, int(encoding_level/2), temp[0:int(encoding_level/2)])
        x2 = decoding(a0, b0, int(encoding_level/2), temp[int(encoding_level/2):encoding_level])
        best_value = np.array([x1, x2])
    generation_best.append(gen_best)    
    generation_worst.append(min(gen_evaluation)[0])
    generation_avg.append(sum(gen_evaluation)/total_pop_size)
    print('Iteration No : ', cnt+1,';', 'Generation Best Value: ', gen_best)
    
    # Mutation
    population = Mutation(new_pop, encoding_level, 0.1)
    cnt += 1
kaka = '=================='
print(7*kaka)
print(r'Genetic Algorithm: total population = %i, encoding level = %i, max iter = %i, search space = [%i, %i]' %(total_pop_size, encoding_level,max_iter,a0,b0))
print(r"Obtained Best Value in %i Iterations is %f " %(cnt, gbest), ',', 'best X* is : ', best_value)
print(7*kaka)

fig, axes = plt.subplots(2,2)
axes[0,0].plot(np.arange(0, len(generation_avg)), generation_best, 'k-', label = 'Gen. Best')
axes[0,1].plot(np.arange(0, len(generation_avg)), generation_avg, 'b--', label = 'Gen. Average')
axes[1,0].plot(np.arange(0, len(generation_avg)), generation_worst, 'r-.', label = 'Gen. Worst')
axes[1,1].plot(np.arange(0, len(generation_avg)), generation_best, 'k-', label = 'Gen. Best')
axes[1,1].plot(np.arange(0, len(generation_avg)), generation_avg, 'b--', label = 'Gen. Average')
axes[1,1].plot(np.arange(0, len(generation_avg)), generation_worst, 'r-.', label = 'Gen. Worst')
#axes[0,0].set_ylim(-10, max(generation_best) + 5)
axes[0,0].legend(loc = 0, ncol = 1)
axes[0,1].legend(loc = 0, ncol = 1)
axes[1,0].legend(loc = 0, ncol = 1)
axes[1,1].legend(loc = 0, ncol = 1)
#plt.tight_layout()
## Make Common X and Y Label
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
# plt.title('Genetic Algorithm: Fitness Function = $1/(f(x) + 100)$', pad = 15)
plt.title('Genetic Algorithm: $Peak$ Function', pad = 15)
plt.xlabel('Generation Number')
plt.ylabel('Objective Value')
#plt.savefig("Results/GA_Given_Prob.eps", dpi=1200, format='eps', orientation='portrait')
#plt.savefig("Results/GA_Peak_Fun.eps", dpi=1200, format='eps', orientation='portrait')
plt.show()
     
stop = timeit.default_timer()
print('Time: ', stop - start) 
    
