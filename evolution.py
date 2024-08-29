from PIL import Image
import numpy as np
import time

from members import Population
from draw import fitnessCalc, voronoiImage


class Evolution:
    def __init__(self, cells, image: Image, num_per_gen=200, gens=100, tournament_size=4, mut_strength=.25,mut_prob=0.0005):

        self.image: Image = image.copy()
        self.image.thumbnail((int(self.image.width / scale), int(self.image.height / scale)), Image.LANCZOS)
        self.image_array = np.asarray(self.image)

        self.cells = cells
        self.gens = gens
        self.mut_strength = mut_strength
        self.seed = np.random.seed(0)

        var_count = cells * 5
        genebounds = [[0, self.image.width] if i % 5 == 0 else [0, self.image.height] if i % 5 == 1 else [0, 256] for i
                      in range(var_count)]

        self.num_per_gen = num_per_gen

        self.genelength = var_count
        self.genebounds = np.array(genebounds, dtype=object)

        self.population = Population(self.num_per_gen, self.genelength)
        self.best_group = None
        self.best_fitness = np.inf
        self.mut_prob = mut_prob

        self.tournament_size = tournament_size

    def update(self, population, first=False):
        source_population = self.population if first else population

        bf_id = np.argmin(source_population.fitnesses)
        best_fitness = source_population.fitnesses[bf_id]

        if best_fitness < self.best_fitness:
            self.best_group = source_population.genes[bf_id, :].copy()
            self.best_fitness = best_fitness


    def mating(self, genes):
        midpoint = len(genes) // 2

        parent_a = genes[:midpoint]
        parent_b = genes[midpoint:]
        offspring = np.empty_like(genes, dtype=int)

        # Alternate between parents for each gene row
        for i in range(len(genes)):
            if np.random.random() >= 0.5:
                offspring[i, :] = parent_a[i % midpoint]

            else:
                offspring[i, :] = parent_b[i % midpoint]


        return offspring

    def mutate(self, genes, limits, mutation_rate=0.0005, mutation_strength=0.25):
        # Create a mask to determine where mutations will occur
        mutation_mask = np.random.choice([True, False], size=genes.shape,
                                         p=[mutation_rate, 1 - mutation_rate])

        # Initialize an array to store variations
        adjusted_genes = np.zeros_like(genes)

        for gene_index in range(genes.shape[1]):
            # Calculate the range of values for the current gene
            gene_range = limits[gene_index][1] - limits[gene_index][0]
            low_bound = -mutation_strength / 2
            high_bound = mutation_strength / 2

            # Generate random variations within the mutation intensity range
            adjusted_genes[:, gene_index] = gene_range * np.random.uniform(low=low_bound, high=high_bound,
                                                                           size=adjusted_genes.shape[0])
            # Apply the variation to the original gene values
            adjusted_genes[:, gene_index] += genes[:, gene_index]

            # Ensure that the new values remain within the specified bounds
            adjusted_genes[:, gene_index] = np.where(adjusted_genes[:, gene_index] > limits[gene_index][1],
                                                     limits[gene_index][1], adjusted_genes[:, gene_index])
            adjusted_genes[:, gene_index] = np.where(adjusted_genes[:, gene_index] < limits[gene_index][0],
                                                     limits[gene_index][0], adjusted_genes[:, gene_index])

        # Convert the adjusted genes to integers
        adjusted_genes = adjusted_genes.astype(int)

        # Apply mutations where the mutation mask is True, otherwise keep the original genes
        mutated_genes = np.where(mutation_mask, adjusted_genes, genes)

        return mutated_genes

    def tournament(self, population, selection_size, tournament_size=4):
        gene_length = population.genes.shape[1]
        winners = Population(selection_size, gene_length)

        total_individuals = len(population.fitnesses)
        selections_per_round = total_individuals // tournament_size
        rounds_needed = selection_size // selections_per_round

        for rounds in range(rounds_needed):
            # Shuffle the population
            population.mix()

            # Reshape fitnesses for tournament comparison
            minitour = population.fitnesses.reshape(-1, tournament_size)
            mini_winners = np.argmin(minitour, axis=1)

            # Convert to absolute indices
            winning_indices = mini_winners + np.arange(0, total_individuals, tournament_size)

            # Select the winning individuals
            start_idx = rounds * selections_per_round
            end_idx = (rounds + 1) * selections_per_round

            winners.genes[start_idx:end_idx, :] = population.genes[winning_indices, :]
            winners.fitnesses[start_idx:end_idx] = population.fitnesses[winning_indices]

        return winners

    def generate(self):
        # Step 1: Create member population by copying and shuffling the current population
        members = Population(self.num_per_gen, self.genelength)
        members.genes[:] = self.population.genes[:]
        members.mix()

        # Step 2: Apply genetic operations (mating and mutation) to the members
        members.genes = self.mating(members.genes)
        members.genes = self.mutate(
            members.genes,
            self.genebounds,
            mutation_rate=self.mut_prob,
            mutation_strength=self.mut_strength
        )

        # Step 3: Evaluate the fitness of the members
        members.fitnesses = fitnessCalc(members.genes, self.image)

        # Step 4: Update the best individual in the population based on the members
        self.update(members)

        # Step 5: Combine the current population with the new members
        self.population.genes = np.vstack((self.population.genes, members.genes))
        self.population.fitnesses = np.concatenate((self.population.fitnesses, members.fitnesses))

        # Step 6: Select the next generation using tournament selection
        self.population = self.tournament(
            self.population,
            self.num_per_gen,
            self.tournament_size
        )

    def action(self):
        # Step 1: Initialize the population with gene bounds and the image array
        self.population.initialize(self.genebounds, self.image_array)

        # Step 2: Evaluate the initial fitness of the population
        self.population.fitnesses = fitnessCalc(self.population.genes, self.image)
        self.update(self.population.fitnesses, first=True)

        # Start the timer
        start_time_seconds = time.time()

        previous_best_fitness = 10000000

        # Step 3: Run the genetic algorithm for the specified number of generations
        for i_gen in range(self.gens + 1):
            self.generate()  # Generate the next population

            # Calculate elapsed time
            elapsed = time.time() - start_time_seconds
            # Calculate fitness improvement
            fitness_improvement = previous_best_fitness - np.mean(self.population.fitnesses)

            # Print the generation summary with fitness improvement
            print(f'Generation: {i_gen}, Best Fitness: {self.best_fitness:.2f}, '
                  f'Avg. Fitness: {np.mean(self.population.fitnesses):.2f}, '
                  f'Fitness Improvement: {fitness_improvement:.2f}, '
                  f'Elapsed Time (sec): {elapsed:.2f}')

            previous_best_fitness = np.mean(self.population.fitnesses)



            if len(np.unique(self.population.genes, axis=0)) < 2:
                # Step 4: Draw and save the final image based on the best individual
                final_image = voronoiImage(self.best_group, self.image.width, self.image.height, scale=scale)
                final_image.save(
                    f"./progress/van_gogh_final_{self.num_per_gen}_{self.cells}_{self.gens}_{self.best_fitness}.png")
                break
            if i_gen%100==0:
                # Step 4: Draw and save the final image based on the best individual
                final_image = voronoiImage(self.best_group, self.image.width, self.image.height, scale=scale)
                final_image.save(
                    f"./progress/van_gogh_final_{self.num_per_gen}_{self.cells}_{self.gens}_{self.best_fitness}.png")







scale = 6

image = Image.open("img/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg").convert('RGB')

runner = Evolution(500, image, num_per_gen=360, gens=30000, tournament_size=20, mut_strength=.65,mut_prob=0.00075)
runner.action()
