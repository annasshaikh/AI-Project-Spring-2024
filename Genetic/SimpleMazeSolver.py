from queue import Queue
import random
import numpy as np
import pygame
from pygame.locals import *
from scipy.ndimage import label
import math

lifespan = 100000

class MapGenerator:
    def __init__(self, width, height, initial_density=0.3, iterations=8):
        self.width = width
        self.height = height
        self.initial_density = initial_density
        self.iterations = iterations
        self.map_array = self.generate_map()
        self.map_array = self.scale_array(self.map_array, 1)
        largest_floor_area = self.find_largest_floor_area(self.map_array)
        self.map_array[~largest_floor_area] = 1  # Convert smaller areas to walls
        self.end = self.find_random_point()
        self.start = self.find_random_point()
        self.definePoints()
        self.distance = self.bfs_with_distance(self.end)
        self.distanceFromStart = self.getdistance(self.start)

    def getdistance(self, start_position):
        return self.distance[start_position[0], start_position[1]]

    def definePoints(self):
        self.map_array[self.end[0], self.end[1]] = 3

    def find_random_point(self):
        while True:
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)
            if self.map_array[x, y] == 0:
                return (x, y)

    def bfs_with_distance(self, start):
        print("Applying BFS")
        height, width = self.map_array.shape
        visited = np.zeros((height, width), dtype=bool)
        distance = np.zeros((height, width), dtype=int)
        queue = Queue()
        queue.put(start)
        visited[start[0]][start[1]] = True

        while not queue.empty():
            current = queue.get()
            neighbors = [(current[0] + 1, current[1]), (current[0] - 1, current[1]),
                         (current[0], current[1] + 1), (current[0], current[1] - 1)]
            for neighbor in neighbors:
                if self.map_array[neighbor[0]][neighbor[1]] == 0 and not visited[neighbor[0]][neighbor[1]]:
                    queue.put(neighbor)
                    visited[neighbor[0]][neighbor[1]] = True
                    distance[neighbor[0]][neighbor[1]] = distance[current[0]][current[1]] + 1
        print(distance)
        return distance

    def scale_array(self, array, factor):
        olen = len(array)
        nlen = olen * factor
        new_array = np.zeros((nlen, nlen))
        for i in range(olen):
            for j in range(olen):
                for k in range(factor):
                    for p in range(factor):
                        if (i * factor + k == 0 or j * factor + p == 0 or i * factor + k == nlen - 1 or j * factor + p == nlen - 1):
                            new_array[i * factor + k][j * factor + p] = 1
                        else:
                            new_array[i * factor + k][j * factor + p] = array[i][j]
        return new_array

    def generate_map(self):
        map_array = np.random.choice([1, 0], size=(self.height, self.width), p=[self.initial_density, 1 - self.initial_density])
        for _ in range(self.iterations):
            map_array = self.apply_rule(map_array)
        return map_array

    def apply_rule(self, map_array, birth_threshold=4, survival_threshold=3):
        height, width = map_array.shape
        new_map = np.zeros_like(map_array)
        for i in range(height):
            for j in range(width):
                alive_neighbors = np.sum(map_array[max(0, i - 1):min(height, i + 2), max(0, j - 1):min(width, j + 2)]) - map_array[i, j]
                if map_array[i, j] == 0:  # If cell is dead
                    if alive_neighbors >= birth_threshold:
                        new_map[i, j] = 1
                else:  # If cell is alive
                    if alive_neighbors >= survival_threshold:
                        new_map[i, j] = 1
        return new_map

    def find_largest_floor_area(self, map_array):
        labeled_map, num_features = self.label_connected_components(map_array == 0)
        sizes = [np.sum(labeled_map == i) for i in range(1, num_features + 1)]
        largest_area_label = np.argmax(sizes) + 1  # Adding 1 because labels start from 1
        return labeled_map == largest_area_label

    def label_connected_components(self, map_array):
        labeled_map, num_features = label(map_array)
        return labeled_map, num_features

def randomVector():
    directions = [-1, 0, 1]
    return [random.choice(directions), random.choice(directions)]

def setMag(vector, mag):
    vector[0] *= mag
    vector[1] *= mag

def addVector(vector1, vector2):
    return [vector1[0] + vector2[0], vector1[1] + vector2[1]]

class DNA:
    def __init__(self, genes=None):
        if genes:
            self.genes = genes
        else:
            self.genes = []
            for i in range(lifespan):
                self.genes.append(randomVector())

    def crossOver(self, partner):
        newGenes = [None] * lifespan
        mid = random.randint(0, len(self.genes) - 1)
        for i in range(lifespan):
            if i <= mid:
                newGenes[i] = self.genes[i]
            else:
                newGenes[i] = partner.genes[i]
        return DNA(newGenes)

    def mutate(self):
        for i in range(len(self.genes)):
            if random.random() < 0.035:
                self.genes[i] = randomVector()

class Population(object):
    """ Creates a array of Players and Performs Functions as a whole"""
    def __init__(self,map):

        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((1000,700))
        pygame.display.set_caption("Game")
        self.font = pygame.font.Font(None, 36)
        
        self.map = map

        self.players = []
# Array to store rocket objects
        self.pop_max = 25  # Maximum Population size

        self.mating_pool = [] # to store copies of rocket objects for selection

        # Creates given amount of objects and stores it in an array
        for _ in range(self.pop_max):
            self.players.append(Player(None,map))

    def run(self):
        """ Updates our Population object params and shows it on screen """
        self.screen.fill((255, 255, 255))
        for span in range(lifespan):
            anyOneAlive = False
            for i in range(self.pop_max):
                if (self.players[i].alive):
                    self.players[i].update()
                    anyOneAlive = True
            if (not anyOneAlive):
                break
            else:
                self.render(self.screen)
        
    def render(self, screen):
        """ Render the full map and red dots for each player's position """
        # Render the map
        # Determine the range of map coordinates to display on the screen
        min_row = 0
        max_row = self.map.height
        min_col = 0
        max_col = self.map.width

        # Draw walls
        for y in range(min_row, max_row):
            for x in range(min_col, max_col):
                if self.map.map_array[y, x] == 1:
                    pygame.draw.rect(screen, (0, 0, 0), (x * 10, y * 10, 10, 10))
                elif self.map.map_array[y, x] == 3:
                    pygame.draw.circle(screen, (255, 225, 0), (x * 10 + 5, y * 10 + 5), 5)
        
         # Render red dots for each player's position
        for player in self.players:
            if player.alive:
                pygame.draw.circle(self.screen, (255, 0, 0), ((player.pos[1]- min_col) * 10 + 5, (player.pos[0]- min_row) * 10 + 5), 5)

        pygame.display.flip()
        self.clock.tick(360)

       

    def evaluate(self):
        """ Evaluates our Population based on some fitness value and creates a mating pool based on them """
        max_fitness = 0
        second_max_fitness = 0
        max_player = None
        second_max_player = None
        sum_fitness = 0

        for i in range(self.pop_max):
            self.players[i].calcFitness()
            if self.players[i].fitness > max_fitness:
                second_max_fitness = max_fitness
                second_max_player = max_player
                max_fitness = self.players[i].fitness
                max_player = self.players[i]
            elif self.players[i].fitness > second_max_fitness:
                second_max_fitness = self.players[i].fitness
                second_max_player = self.players[i]
            sum_fitness += self.players[i].fitness

        print("Average Fitness: ", sum_fitness / max_fitness)

        for i in range(self.pop_max):
            self.players[i].fitness /= max_fitness

        self.mating_pool = []

        for i in range(self.pop_max):
            n = math.floor(self.players[i].fitness * 100)
            for _ in range(n):
                self.mating_pool.append(self.players[i])
        print("Mating Pool: ", len(self.mating_pool))

        
        return max_player, second_max_player

    def natural_selection(self,max_player, second_max_player):
        """ Selects two fit members of Population and performs crossover and mutation on them, 
            then makes the new population based on the new created member """
        new_players = []
        new_players.append(max_player)
        new_players.append(second_max_player)
        for _ in range(len(self.players) - 2):
            parentA = np.random.choice(self.mating_pool).dna
            parentB = np.random.choice(self.mating_pool).dna
            childDNA = parentA.crossOver(parentB)
            childDNA.mutate()
            new_players.append(Player(childDNA,self.map))
        self.players = new_players


class Player:
    def __init__(self, dna=None, map=None):
        if dna:
            self.dna = dna
        else: 
            self.dna = DNA()
        self.map = map
        self.pos = map.start
        self.genes = self.dna.genes
        self.alive = True
        self.crashed = False
        self.completed = False
        self.count = 0
        self.fitness = 0

    def isOnWall(self,point):
        return self.map.map_array[point[0], point[1]] == 1

    def isCompleted(self,point):
        return self.map.map_array[point[0], point[1]] == 3

    def update(self):
        NewPos = addVector(self.pos,self.genes[self.count])
        if (self.isOnWall(NewPos)):
            self.crashed = True
            self.alive = False
        if self.isCompleted(NewPos):
            self.completed = True
            self.alive = False
        if not self.crashed and not self.completed:
            self.pos = NewPos
            self.count += 1

    def render(self, screen):
        if not self.completed:
            pygame.draw.circle(screen, (255, 0, 0), (self.pos[1], self.pos[0]), 1)  # Draw a red dot at player position
        else:
            pygame.draw.circle(screen, (255, 255, 0), (self.pos[1], self.pos[0]), 1)  # Draw a yellow dot at player position

    def calcFitness(self):
        fitness = self.map.distanceFromStart - self.map.getdistance(self.pos)
        if self.completed:
            fitness *= 10
        if self.crashed:
            fitness /= 10
        self.fitness = fitness

    def crossover(self, ParentB):
        NewDNA = self.dna.crossOver(ParentB)
        return Player(NewDNA, self.map)

    def mutation(self):
        self.dna.mutate()

def main():
   
    map_generator = MapGenerator(60, 60)
    population = Population(map_generator)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        print("Runing")

        population.run()
        a,b = population.evaluate()
        population.natural_selection(a,b)

    pygame.quit()

if __name__ == "__main__":
    main()
