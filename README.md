# Simple Neural network with genetic Algorithm


[![CircleCI](https://circleci.com/gh/edwardmartinsjr/simple-genetic-algorithm/tree/master.svg?style=shield)](https://circleci.com/gh/edwardmartinsjr/simple-neural-net-with-genetic-algorithm/tree/master)

1. This program is a single .py file.

2. This program is written in python 3.8, using only python’s built-in libraries.

3. This program contains a main() method that try to match a pre-defined optimal solution:
	
    a. Start Pong game;
	
    b. Starting population;

    c. Iterate over the generations of genetic algorithm:
	
    - Apply NN for each individual and scores the turn;
    - Exit loop if maximum generation is reached.

4. Install requirements:
```
pip install -r requirements.txt
```

5. Test:
```
python test_main.py -v
```

6. Run:
```
python main.py
```
