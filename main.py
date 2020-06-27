import random
import numpy as np
import matplotlib.pyplot as plt
import logging
from pynput.keyboard import Key, Controller
import time
import csv
from threading import Thread

l_score = 0

keyboard = Controller()  # Create the controller

# create logger
logger = logging.getLogger('simple-neural-net-with-genetic-algorithm')
logger.setLevel(logging.INFO)

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

    logger.info(f"NN values: [{measurements, individual, final_output}]")
    
    return final_output

def create_starting_population(individuals, chromosome_length, weight_low, weight_high):
    # Set up an initial array of values between weight_low and weight_high
    population = np.random.uniform(weight_low,weight_high,(individuals, chromosome_length))
    
    return population

def get_measurements():
    measurements = [0,0]

    with open('game_report.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                measurements = np.array([int(row["ball_pos_c"]),int(row["ball_pos_l"])])
            except:
                measurements = [0,0]

    return measurements

def get_game_status():
    global l_score
    score = 0

    with open('game_report.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                score = int(row["l_score"])
            except:
                score = l_score

    
    if score > l_score:
        l_score = score
        return False
    else: 
        return True

def start_game():
    import  game
    game.init()
    game.track_event()

def play(individual, bias, player, generation):
    score = 0
    turn = 0
    
    while(True):
        print(f'Player: {player}, Score: {score}, Generation: {generation}')
        turn += 1

        # Get game measurements
        measurements = get_measurements()

        # Get NN output
        final_output = get_nn_output(individual, bias, measurements)
    
        # Put neural network (as a player) against an opponent
        if final_output > 0 and final_output < 0.5:
            keyboard.press(Key.up)
            time.sleep(0.100)
            keyboard.release(Key.up)
        if final_output >= 0.5 and final_output < 0.9:
            keyboard.press(Key.down)
            time.sleep(0.100)
            keyboard.release(Key.down)

        # Randomly move oponent
        if bool(random.getrandbits(1)):
            keyboard.press('z')
            time.sleep(0.100)
            keyboard.release('z')
        else:
            keyboard.press('s')
            time.sleep(0.100)
            keyboard.release('s')

        # Get action report
        if get_game_status():
            score += 1
        else:
            print('Game over')
            break
        

    return score

# Calculate the neural network fitness based on performance against an opponent
def calculate_neural_net_fitness(population, bias, generation):
    scores = []
    count = 0
    
    # For each individual create a neural network with three neurons in the hidden layer and two measurements
    for individual in population:
        count += 1
        scores.append(play(individual, bias, count, generation))

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
    population_size = 10
    maximum_generation = 5
    best_score_progress = [] # Tracks progress
    weight_low = -1
    weight_high = 1
    mutation_rate = 0.002
    bias = 0.001
    count_generation = 1

    # Start Game
    t1 = Thread(target = start_game)
    t1.start()

    # just clean game status and score
    with open('game_report.csv', mode='w') as csv_file:
        fieldnames = ['ball_pos_l', 'ball_pos_c', 'r_score', 'l_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'ball_pos_c': 0, 'ball_pos_l': 0, 'r_score': 0, 'l_score': 0})

    # Create starting population
    population = create_starting_population(population_size, chromosome_length, weight_low, weight_high)

    # Display best score in starting population
    scores = calculate_neural_net_fitness(population, bias, count_generation)
    best_score = np.max(scores)
    print ('Starting best score, successful frames: ',best_score)

    # Add starting best score to progress tracker
    best_score_progress.append(best_score)
    
    # Now we'll go through the generations of genetic algorithm
    for generation in range(maximum_generation):
        count_generation += 1
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
        scores = calculate_neural_net_fitness(population, bias, count_generation)
        best_score = np.max(scores)
        best_score_progress.append(best_score)

    # GA has completed required generation
    print ('End best score, successful frames: ', np.max(best_score_progress))
    print ('Total generations: ', maximum_generation)

    # Plot progress
    plt.plot(best_score_progress)
    plt.xlabel('Generation')
    plt.ylabel('Best score (% target)')
    plt.show()