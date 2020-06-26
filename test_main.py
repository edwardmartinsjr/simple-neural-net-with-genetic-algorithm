import unittest
from unittest.mock import patch
import numpy as np
import main

# START TESTS
class Test(unittest.TestCase):
    def test_sigmoid(self):
        ''' Test sigmoid calc '''

        x = 1
        sigmoid_formula = 1/(1 + np.exp(-x))
        self.assertEqual(main.sigmoid(x), sigmoid_formula)

    def test_1_NN(self):
        ''' 
        Test one single neuron and two measurements
        |   INPUT LAYER  |HIDDEN LAYER|OUTPUT LAYER|
        m1 --- (w1m1) ---|
                         |----> N1 ---> Output 1
        m2 --- (w1m2) ---|
        '''

        m1 = 1.5 
        m2 = 5.1
        w1m1 = 1
        w1m2 = 2

        bias = 1.5

        measurement = [m1, m2]
        weight = [w1m1, w1m2]

        o1 = main.NN(measurement, weight, bias)

        x = (m1*w1m1)+(m2*w1m2)+bias
        sigmoid_formula = 1/(1 + np.exp(-x))

        self.assertEqual(o1, sigmoid_formula)

    def test_2_NN(self):
        ''' 
        Test two neurons in the hidden layer and two measurements
        |   INPUT LAYER  |       HIDDEN LAYER       |             OUTPUT LAYER            |
        m1 --- (w1m1) ---|
                         |----> N1 ---> Output 1 ---|
        m2 --- (w1m2) ---|                          |--- (w1o1) ---|
                                                                   |----> N3 ---> Output 3
        m1 --- (w2m1) ---|                          |--- (w1o2) ---|        
                         |----> N2 ---> Output 2 ---|
        m2 --- (w2m2) ---|
        '''

        m1 = 1.5 
        m2 = 5.1
        w1m1 = 1
        w1m2 = 2
        w2m1 = 3
        w2m2 = 4
        w1o1 = 5
        w1o2 = 6

        bias = 1.5

        measurement = [m1, m2]
        
        weight = [w1m1, w1m2]
        o1 = main.NN(measurement, weight, bias)
        
        weight = [w2m1, w2m2]
        o2 = main.NN(measurement, weight, bias)
        
        measurement = [o1, o2]
        weight = [w1o1, w1o2]
        o3 = main.NN(measurement, weight, bias)
        
        o1 = 1/(1 + np.exp(-((m1*w1m1)+(m2*w1m2)+bias)))
        o2 = 1/(1 + np.exp(-((m1*w2m1)+(m2*w2m2)+bias)))
        x =  1/(1 + np.exp(-((o1*w1o1)+(o2*w1o2)+bias)))

        self.assertEqual(o3, x)

    def test_3_NN(self):
        ''' 
        Test three neurons in the hidden layer and two measurements
        |   INPUT LAYER  |       HIDDEN LAYER       |             OUTPUT LAYER            |
        m1 --- (w1m1) ---|
                         |----> N1 ---> Output 1 ---|
        m2 --- (w1m2) ---|                          |--- (w1o1) ---|
                                                                   |
        m1 --- (w2m1) ---|                                         |
                         |----> N2 ---> Output 2 ---|--- (w1o2) ---|----> Output 3        
        m2 --- (w2m2) ---|                                         |
                                                                   |
        m1 --- (w3m1) ---|                          |--- (w1o3) ---|        
                         |----> N3 ---> Output 3 ---|
        m2 --- (w3m2) ---|
        ''' 

        m1 = 1.5 
        m2 = 5.1
        w1m1 = 1
        w1m2 = 2
        w2m1 = 3
        w2m2 = 4
        w3m1 = 5
        w3m2 = 6
        w1o1 = 7
        w1o2 = 8
        w1o3 = 9

        bias = 1.5

        measurement = [m1, m2]
        
        weight = [w1m1, w1m2]
        o1 = main.NN(measurement, weight, bias)
        
        weight = [w2m1, w2m2]
        o2 = main.NN(measurement, weight, bias)

        weight = [w3m1, w3m2]
        o3 = main.NN(measurement, weight, bias)        
        
        measurement = [o1, o2, o3]
        weight = [w1o1, w1o2, w1o3]
        o4 = main.NN(measurement, weight, bias)
        
        o1 = 1/(1 + np.exp(-((m1*w1m1)+(m2*w1m2)+bias)))
        o2 = 1/(1 + np.exp(-((m1*w2m1)+(m2*w2m2)+bias)))
        o3 = 1/(1 + np.exp(-((m1*w3m1)+(m2*w3m2)+bias)))
        x =  1/(1 + np.exp(-((o1*w1o1)+(o2*w1o2)+(o3*w1o3)+bias)))

        self.assertEqual(o4, x)

    def test_create_starting_population(self):
        ''' 
        Test a starting population creation (equivalent weights for three neurons in the hidden layer and two measurements)
        ''' 

        population_size = 1
        chromosome_length = 9
        weight_low = 0
        weight_high = 1000

        starting_population = main.create_starting_population(population_size, chromosome_length, weight_low, weight_high)
        
        self.assertEqual(len(starting_population), population_size)
        self.assertEqual(len(starting_population[0]), chromosome_length)

    def test_calculate_neural_net_fitness(self):
        ''' 
        Test the neural network fitness based on performance against an opponent
        '''
        population = [range(9),range(9),range(9)]
        bias = 1

        scores = main.calculate_neural_net_fitness(population, bias)
        self.assertEqual(len(scores), 3)

    def test_select_individual_by_tournament(self):
        '''
        Test individual tournament selection
        '''

        population_size = 3
        chromosome_length = 9
        weight_low = 0
        weight_high = 1000
        
        # Create starting population
        population = main.create_starting_population(population_size, chromosome_length, weight_low, weight_high)
        bias = 1

        # Best score in starting population
        scores =  main.calculate_neural_net_fitness(population, bias)
        
        self.assertEqual(len(main.select_individual_by_tournament(population,scores)), 9)

    def test_breed_by_crossover(self):
        '''
        Test individual generation by crossover
        '''

        population_size = 3
        chromosome_length = 9
        weight_low = 0
        weight_high = 1000
        bias = 1
        
        # Create starting population
        population = main.create_starting_population(population_size, chromosome_length, weight_low, weight_high)
        reference = population[0]

        # Best score in starting population
        scores =  main.calculate_neural_net_fitness(population, bias)

        # Create children by parent crossover
        parent_1 = main.select_individual_by_tournament(population,scores)
        parent_2 = main.select_individual_by_tournament(population,scores)

        self.assertEqual(len(main.breed_by_crossover(parent_1, parent_2)), 2)

    def test_randomly_mutate_population(self):
        '''
        Test individual generation by crossover
        '''

        population_size = 1
        chromosome_length = 9
        weight_low = 0
        weight_high = 1000
        
        # Create starting population
        population = main.create_starting_population(population_size, chromosome_length, weight_low, weight_high)

        # Apply mutation
        mutation_rate = 0.002
        self.assertEqual(len(main.randomly_mutate_population(population, mutation_rate)), population_size)

    def test_create_reference_solution(self):
        '''
        Test reference_solution creation
        '''

        chromosome_length = 9
        weight_low = 0
        weight_high = 1000

        self.assertEqual(len(main.create_reference_solution(chromosome_length, weight_low, weight_high)), 9)

    def test_get_nn_output(self):
        '''
        Test getting nn output
        '''
        self.assertGreater(main.get_nn_output(range(9),1,[1,1]), 0)

    def test_play(self):
        '''
        Test getting nn output
        '''
        self.assertGreaterEqual(main.play(range(9),1), 0)


    

    # play



unittest.main()


