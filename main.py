import random
import numpy as np
import matplotlib.pyplot as plt

# Feed Forward
# Calculate the neural network measurement, weight, and bias
def NN(measurement, weight, bias):
    z = np.sum(np.multiply(measurement, weight)) + bias
    return sigmoid(z)

# Calculate the intermediate value
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def create_reference_solution(chromosome_length, weight_low, weight_high):
    # Set up an reference of values between weight_low and weight_high
    reference = np.random.uniform(weight_low,weight_high,chromosome_length)
    
    return reference

def get_nn_output(individual, bias, measurements):
    weight = [individual[0], individual[1]]
    n1_output = NN(measurements, weight, bias)
    
    weight = [individual[2], individual[3]]
    n2_output = NN(measurements, weight, bias)

    weight = [individual[4], individual[5]]
    n3_output = NN(measurements, weight, bias)        
    
    output_weight = [individual[6], individual[7], individual[8]]
    output_measurements = [n1_output, n2_output, n3_output]
    final_output = NN(output_measurements, output_weight, bias)

    print(measurements, individual, final_output)

    return final_output

def create_starting_population(individuals, chromosome_length, weight_low, weight_high):
    # Set up an initial array of values between weight_low and weight_high
    population = np.random.uniform(weight_low,weight_high,(individuals, chromosome_length))
    
    return population

def play(individual, bias):
    score = 0

    # Start game
    while(True):
        # TODO: Implement game measurements getting
        # Get game measurements
        measurements = [np.random.uniform(0,1), np.random.uniform(0,1)]

        # Get NN output
        final_output = get_nn_output(individual, bias, measurements)
    
        # Put neural network against an opponent
        if final_output > 0 and final_output < 0.5:
            print('left')
        if final_output > 0.5 and final_output < 0.9:
            print('right')

        # Apply action
        # TODO: Implement Apply action

        # Get action report
        if bool(random.getrandbits(1)):
            score += 1
        else:
            print('Dead')
            break

    return score

# Calculate the neural network fitness based on performance against an opponent
def calculate_neural_net_fitness(population, bias):
    scores = []
    
    # For each individual create a neural network with three neurons in the hidden layer and two measurements
    for individual in population:
        scores.append(play(individual, bias))

    return scores


def select_individual_by_tournament(population, scores):
    # Get population size
    population_size = len(scores)
    
    # Pick individuals for tournament
    fighter_1 = np.random.randint(0, population_size-1)
    fighter_2 = np.random.randint(0, population_size-1)
    
    # Get fitness score for each
    fighter_1_fitness = scores[fighter_1]
    fighter_2_fitness = scores[fighter_2]
    
    # Identify undividual with highest fitness
    # Fighter 1 will win if score are equal
    if fighter_1_fitness >= fighter_2_fitness:
        winner = fighter_1
    else:
        winner = fighter_2
    
    # Return the chromsome of the winner
    return population[winner, :]


def breed_by_crossover(parent_1, parent_2):
    # Get length of chromosome
    chromosome_length = len(parent_1)
    
    # Pick crossover point, avoding ends of chromsome
    crossover_point = np.random.randint(1,chromosome_length-1)
    
    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                        parent_2[crossover_point:]))
    
    child_2 = np.hstack((parent_2[0:crossover_point],
                        parent_1[crossover_point:]))
    
    # Return children
    return child_1, child_2
    

def randomly_mutate_population(population, mutation_probability):
    
    # Apply random mutation
        random_mutation_array = np.random.random(
            size=(population.shape))
        
        random_mutation_boolean = \
            random_mutation_array <= mutation_probability

        population[random_mutation_boolean] = \
        np.logical_not(population[random_mutation_boolean])
        
        # Return mutation population
        return population
    

# *************************************
# ******** MAIN ALGORITHM CODE ********
# *************************************
if __name__== '__main__':
    # Set general parameters
    chromosome_length = 9
    population_size = 500
    maximum_generation = 10
    best_score_progress = [] # Tracks progress
    gen_count =  1 # register the number of the generations
    weight_low = -1
    weight_high = 1
    mutation_rate = 0.002
    bias = 0.001    

    # Create starting population
    population = create_starting_population(population_size, chromosome_length, weight_low, weight_high)

    # Display best score in starting population
    scores = calculate_neural_net_fitness(population, bias)
    best_score = np.max(scores)
    print ('Starting best score, successful frames: ',best_score)

    # Add starting best score to progress tracker
    best_score_progress.append(best_score)
    
    # Now we'll go through the generations of genetic algorithm
    for generation in range(maximum_generation-1):
        # Create an empty list for new population
        new_population = []
        
        # Create new popualtion generating two children at a time
        for i in range(int(population_size/2)):
            parent_1 = select_individual_by_tournament(population, scores)
            parent_2 = select_individual_by_tournament(population, scores)
            child_1, child_2 = breed_by_crossover(parent_1, parent_2)
            new_population.append(child_1)
            new_population.append(child_2)
        
        # Replace the old population with the new one
        population = np.array(new_population)

        # Apply mutation
        population = randomly_mutate_population(population, mutation_rate)
        
        # Score best solution, and add to tracker
        scores = calculate_neural_net_fitness(population, bias)
        best_score = np.max(scores)
        best_score_progress.append(best_score)

        # Increment generation counter 
        gen_count += 1

    # GA has completed required generation
    print ('End best score, successful frames: ',best_score)
    print ('Total generations: ', gen_count)

    # Plot progress
    plt.plot(best_score_progress)
    plt.xlabel('Generation')
    plt.ylabel('Best score (% target)')
    plt.show()