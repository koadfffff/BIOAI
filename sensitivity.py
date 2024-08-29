import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from evolution import Evolution

mutstrengths = [0.48,0.56,0.64,0.72,0.90,0.98]
mutprobs = [0.00005,0.0001,0.0005,0.001,0.005]
popsizes = [100,200,300,400,500,600]
toursizes = [4,8,10,16,20]
cellsls = [100,200,300,400,500]

list = cellsls

totaltime=[]
totalfitness=[]
totalgen=[]

for i in list:
    scale = 6

    image = Image.open("img/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg").convert('RGB')

    cells = i
    popsize = 400
    genbud = 1500
    toursize = 20
    mutstr = 0.65
    mutprob = 0.00075

    best_fitness = np.inf

    runner = Evolution(cells, image, num_per_gen=popsize, gens=genbud, tournament_size=toursize, mut_strength=mutstr, mut_prob=mutprob)
    times = []
    fitnesses = []
    gens=[]
    for elapsed,gen in runner.action():
        times.append(elapsed)
        fitnesses.append(runner.best_fitness)
        gens.append(gen)
        # You might want to check the fitness condition here if applicable
        # if runner.best_fitness < 30000:
        #     break
    totalgen.append(gens)
    totaltime.append(times)
    totalfitness.append(fitnesses)

for i in range(len(cellsls)):
    plt.plot(totalgen[i],totalfitness[i],label='Number of Cells: '+ str(cellsls[i]))
plt.legend()
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.savefig('sensitivityimg/cells.png')
plt.show()

