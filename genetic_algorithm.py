import random
import numpy as np
from neural_network import *
import csv
import time
import math


# Funzione per l'addestramento
def fitness(individual, epochs):
    # Addestra la CNN con i parametri dell'individuo 
    num_conv_layers, conv_size, num_fc_layers, fc_size = individual
    cnn_model = create_cnn(num_conv_layers, conv_size, num_fc_layers, fc_size)
    start_time = time.time()
    accuracy, history = train_cnn(cnn_model, epochs)
    training_time = time.time() - start_time;
    fitness_score = accuracy / math.log10(training_time)  # La fitness è data dall'accuracy fratto il log10 del tempo di addestramento
    return accuracy, fitness_score, training_time, history, individual

# Inizializzazione della popolazione
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        individual = [
            random.randint(1, 5),  # Numero di strati convoluzionali
            random.randint(16, 256),  # Dimensione degli strati convoluzionali
            random.randint(1, 3),  # Numero di strati fully-connected
            random.randint(16, 256)  # Dimensione degli strati fully-connected
        ]
        population.append(individual)
    return population

# Selezione dei genitori basata sulla roulette wheel
def select_parents(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fit / total_fitness for fit in fitness_values] # Calcolo delle probabilità per ogni individuo
    selected_indices = np.random.choice(len(population), size=2, p=probabilities) # Scelta di due indici in base alla probabilità
    return [population[i] for i in selected_indices] # Estrazione degli individui

# Crossover a un punto
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1) # Crossover a un punto casuale
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutazione
def mutate(mutated_individual, mutation_rate=0.1):
    for i in range(len(mutated_individual)): # Cicla gene per gene
        if random.random() < mutation_rate: # Mutation rate del 10%
            mutated_individual[i] = random.randint(1, 5) if i % 2 == 0 else random.randint(16, 256) # Mutazione dell'individuo
            # Se l'indice è pari allora il gene corrisponde al numero di strati convoluzionali o fully-connected
            # Altrimenti corrisponde alla dimensione dello strato
    return mutated_individual

def save_individual(filename, generation, individual, accuracy, fitness_score, training_time):
    header = ["Generation", "NumConvLayers", "ConvSize", "NumFCLayers", "FCSize", "Accuracy", "Fitness", "Training time"] # Intestazione
    data_to_write = [generation] + individual + [accuracy, fitness_score, training_time] # Si assicura che sia una lista
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0: # Se il file è vuoto allora scrivi l'intestazione
            writer.writerow(header)
        writer.writerow(map(str, data_to_write)) # Converte ogni elemento in una stringa

population_size = 20
target_fitness = 0.33  # Fitness desiderata
population = initialize_population(population_size)
epochs = 10

population_filename = 'population.csv'
best_individual = None
best_accuracy = 0.0
best_fitness = 0.0

total_time = time.time();
# Ciclo fino a trovare un individuo con la fitness desiderata
generation = 0

while best_fitness < target_fitness:
    fitness_values = [fitness(individual, epochs) for individual in population]

    for fit in fitness_values:
        accuracy, fitness_score, training_time, _, individual = fit  # Estrai i valori restituiti dalla fitness function
        save_individual(population_filename, generation, individual, accuracy, fitness_score, training_time)
    
    accuracies = [fit[0] for fit in fitness_values]
    fitness_scores = [fit[1] for fit in fitness_values]
    training_times = [fit[2] for fit in fitness_values]
    history = [fit[3] for fit in fitness_values];
    
    # Identifica il miglior individuo nella popolazione corrente
    best_index = np.argmax(fitness_scores) if fitness_scores else None

    if best_index is not None:
        best_individual = population[best_index] # Selezione del miglior individuo
        best_accuracy = accuracies[best_index] # Selezione dell' accuracy associata all'individuo
        best_fitness = fitness_scores[best_index] # Selezione della fitness associata all'individuo
        best_time = training_times[best_index] # Selezione del tempo di training del miglior individuo

        print(f'Generazione {generation + 1}:  Miglior fitness = {best_fitness}, Accuracy = {best_accuracy} Parametri = {best_individual}')
    else:
        print(f'Generazione {generation + 1}: Nessun individuo valido nella popolazione')

    # Selezione, crossover e mutazione per generare la nuova popolazione
    new_population = []
    for _ in range(population_size // 2): # Si scelgono 10 genitori
        parent1, parent2 = select_parents(population, fitness_scores)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.extend([child1, child2]) 

    population = new_population

    generation += 1

print(f'Terminato dopo {generation} generazioni in {time.time()-total_time}. Individuo con la fitness desiderata: {best_individual}, Accuracy: {best_accuracy}, Fitness: {best_fitness}, addestrato in {best_time} secondi')
