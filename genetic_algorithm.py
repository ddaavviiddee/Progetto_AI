import random
import numpy as np
from neural_network import create_cnn, train_cnn
from copy import deepcopy
import csv
import time


# Funzione di fitness
def fitness(individual):
    # Addestra la CNN con i parametri dell'individuo e restituisci l'accuracy
    print("Training with individual")
    num_conv_layers, conv_size, num_fc_layers, fc_size = individual
    cnn_model = create_cnn(num_conv_layers, conv_size, num_fc_layers, fc_size)
    accuracy = train_cnn(cnn_model, epochs=10)
    return accuracy

# Inizializzazione della popolazione
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        individual = [
            random.randint(1, 5),  # Numero di strati convoluzionali
            random.randint(16, 64),  # Dimensione degli strati convoluzionali
            random.randint(1, 3),  # Numero di strati fully-connected
            random.randint(16, 256)  # Dimensione degli strati fully-connected
        ]
        population.append(individual)
    return population

# Selezione dei genitori basata sulla roulette wheel
def select_parents(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fit / total_fitness for fit in fitness_values]
    selected_indices = np.random.choice(len(population), size=2, p=probabilities)
    return [population[i] for i in selected_indices]

# Crossover a un punto
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutazione
def mutate(individual, mutation_rate=0.1):
    mutated_individual = deepcopy(individual)
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = random.randint(1, 5) if i % 2 == 0 else random.randint(16, 256)
    return mutated_individual

# Funzione per salvare la popolazione in un file CSV
def save_individuals(filename, best_individual, best_accuracy):
    header = ["NumConvLayers", "ConvSize", "NumFCLayers", "FCSize", "Accuracy"]
    with open(filename, mode='a', newline='') as file:  # Usa 'a' per appendere al file
        writer = csv.writer(file)
        if file.tell() == 0:  # Scrivi l'header solo se il file è vuoto
            writer.writerow(header)
        # Scrivi i dati del miglior individuo di questa generazione
        data_to_write = list(best_individual) + [best_accuracy]
        writer.writerow(map(str, data_to_write))

        

# Esempio di utilizzo dell'algoritmo genetico
population_size = 20
generations = 10
population = initialize_population(population_size)

# Esempio di utilizzo delle funzioni di salvataggio e caricamento
population_filename = 'population.csv'
best_individual = None
best_accuracy = 0.0

start_time = time.time()

# All'interno del ciclo delle generazioni
for generation in range(generations):

    fitness_values = [fitness(individual) for individual in population]
    # Identifica il miglior individuo nella popolazione corrente
    best_index = np.argmax(fitness_values) if fitness_values else None

    if best_index is not None:
        best_individual = population[best_index]
        best_accuracy = fitness_values[best_index]
        save_individuals(population_filename, best_individual, best_accuracy)
        print(f'Generazione {generation + 1}: Miglior Accuracy = {best_accuracy}, Parametri = {best_individual}')
    else:
        print(f'Generazione {generation + 1}: Nessun individuo valido nella popolazione')

    # Selezione, crossover e mutazione per generare la nuova popolazione
    new_population = []
    for _ in range(population_size // 2):
        parent1, parent2 = select_parents(population, fitness_values)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.extend([child1, child2])

    population = new_population

print(time.time() - start_time)

