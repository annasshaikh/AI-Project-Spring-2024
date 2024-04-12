import os
from queue import Queue
import random
import numpy as np
import pygame
from pygame.locals import *
from scipy.ndimage import label
import math
import copy
import neat
lifespan = 100000
gen = 0
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
        self.enemyList = self.spawn_enemies(1)
        #self.map_array = self.load_map_from_csv("MAP.csv")
        self.distance = self.bfs_with_distance(self.end)
        self.distanceFromStart = self.getdistance(self.start)
        #self.save_map_to_csv(self.map_array,"MAP.csv")
    def save_map_to_csv(self,map_array, file_path):
        map = copy.deepcopy(map_array)
        for e in self.enemyList:
            p = e.position
            map[p[0]][p[1]] = 4
        map[self.start[0]][self.start[1]] = 2
        try:
            np.savetxt(file_path, map, delimiter=',', fmt='%d')
            print("Map saved successfully to:", file_path)
        except Exception as e:
            print("An error occurred while saving the map:", e)

    def load_map_from_csv(self,file_path):
        try:
            map_array = np.loadtxt(file_path, delimiter=',', dtype=int)
            
            start_points = np.where(map_array == 2)
            end_points = np.where(map_array == 3)
            bot_locations = np.where(map_array == 4)

            # Replace specific values with meaningful entities
            map_array[map_array == 2] = 0  # Reset previous starting points to empty space
            map_array[map_array == 3] = 0  # Reset previous end points to empty space
            map_array[map_array == 4] = 0  # Reset previous bot locations to empty space
            
            self.start = [start_points[0][0],start_points[1][0]]
            self.end = [end_points[0][0],end_points[1][0]]
            if len(start_points[0]) != 1 or len(end_points[0]) != 1:
                raise ValueError("There should be exactly one starting point and one end point in the map.")
            
            # Set starting point, end point, and bot locations
            map_array[start_points] = 0
            map_array[end_points] = 3
            map_array[bot_locations] = 0
            for l in bot_locations:
                self.enemyList = [Enemy(map_array,start_points) for _ in range(2)]
            return map_array
        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")
            return None
        except Exception as e:
            print("An error occurred while loading the map:", e)
            return None

    def spawn_enemies(self,number):
        enemylist = [Enemy(self.map_array,self.start) for _ in range(number)]
        return enemylist
    
    def getdistance(self, start_position):
        return self.distance[int(start_position[0])][int(start_position[1])]

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
        print(height,"  ",width)
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

class PopulationRender(object):
    """ Creates a array of Players and Performs Functions as a whole"""
    def __init__(self,map):

        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((1000,700))
        pygame.display.set_caption("Game")
        self.font = pygame.font.Font(None, 36)
        self.map = map

    # def run(self):
    #     """ Updates our Population object params and shows it on screen """
    #     self.screen.fill((255, 255, 255))
    #     for span in range(lifespan):
    #         anyOneAlive = False
    #         for i in range(self.pop_max):
    #             if (self.players[i].alive):
    #                 self.players[i].update()
    #                 anyOneAlive = True
    #         if (not anyOneAlive):
    #             break
    #         else:
    #             self.render(self.screen)
    #         ia = input()
        
    def render(self,players):
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
                    pygame.draw.rect(self.screen, (255, 255, 255), (x * 10, y * 10, 10, 10))
                elif self.map.map_array[y, x] == 3:
                    pygame.draw.circle(self.screen, (255, 225, 0), (x * 10 + 5, y * 10 + 5), 5)
        
            # Render red dots for each player's position
        for player in players:
            if player.alive:
                pygame.draw.circle(self.screen, (255, 0, 0), ((player.pos[1]- min_col) * 10 + 5, (player.pos[0]- min_row) * 10 + 5), 5)
                for enemy in player.enemylist:
                    pygame.draw.circle(self.screen, (255, 0, 225), ((enemy.position[1]- min_col) * 10 + 5, (enemy.position[0]- min_row) * 10 + 5), 5)
                
        pygame.display.flip()
        self.clock.tick(30)       

class Enemy:
    def __init__(self, map_array, player_position,enemyPosition=None):
        self.map_array = map_array
        if enemyPosition:
            self.position = enemyPosition
        else:
            self.position = self.find_random_point()
        self.player_position = player_position
        self.speed = 0.25

    def updatePlayerPosition(self,position):
        self.player_position = position
        self.move_towards_player()
  
    def find_random_point(self):
        while True:
            x = np.random.randint(self.map_array.shape[1])
            y = np.random.randint(self.map_array.shape[0])
            if self.map_array[y, x] == 0:
                return np.array([y, x])
            
    def is_line_of_sight_clear(self):
        dx = self.player_position[1] - self.position[1]
        dy = self.player_position[0] - self.position[0]
        distance = max(abs(dx), abs(dy))
        angle = math.atan2(dy, dx)  # Calculate the angle to the player
        
        if distance == 0:
            return [True, distance, angle]  # Return distance, angle, and line of sight result

        dx /= distance
        dy /= distance

        for i in range(1, int(distance) + 1):
            x = int(self.position[1] + dx * i)
            y = int(self.position[0] + dy * i)
            if x < 0 or y < 0 or x >= self.map_array.shape[1] or y >= self.map_array.shape[0] or self.map_array[y, x] == 1:
                return [False, distance, angle]  # Return distance, angle, and line of sight result
        return [True, distance, angle]  # Return distance, angle, and line of sight result

    def move_towards_player(self):
        if not self.is_line_of_sight_clear()[0]:
            return
        dx = self.player_position[1] - self.position[1]
        dy = self.player_position[0] - self.position[0]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance != 0:
            dx /= distance
            dy /= distance

        new_position = np.array([self.position[0] + dy * self.speed, self.position[1] + dx * self.speed])

        if self.is_valid_move(new_position):
            self.position = new_position

    def is_valid_move(self, new_position):
        x, y = new_position.astype(int)
        if x < 0 or y < 0 or x >= self.map_array.shape[0] or y >= self.map_array.shape[1]:
            return False
        if self.map_array[x, y] == 1:  # If the new position is a wall
            return False
        return True
    
    def isTouching(self):
        dx = self.player_position[1] - self.position[1]
        dy = self.player_position[0] - self.position[0]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return distance < 2

class Player:
    def __init__(self, enemylist, map=None):
        self.map = map
        self.pos = map.start
        self.velocity = [0.5, 0.5]  # Initial velocity
        self.enemylist = copy.deepcopy(enemylist)
        self.alive = True
        self.crashed = False
        self.completed = False
        self.killed = False
        self.count = 0
        self.fitness = 0
        self.sensor_angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4]
        self.sensor_distances = [0] * len(self.sensor_angles)
    def getInputs(self):
        inputs = [] + self.sensor_distances
        for enemy in self.enemylist:
            sight_data = enemy.is_line_of_sight_clear()
            if sight_data[0]:
                inputs = inputs + sight_data[1] + sight_data[2]
            else:
                inputs = inputs + [-1] + [-1]
        return inputs
    def update_sensor_distances(self):
        for i, angle in enumerate(self.sensor_angles):
            dx = math.cos(angle)
            dy = math.sin(angle)
            x = self.pos[0]
            y = self.pos[1]
            distance = 0
            while True:
                x += dx
                y += dy
                if x < 0 or y < 0 or x >= self.map.map_array.shape[1] or y >= self.map.map_array.shape[0] or self.map.map_array[int(y), int(x)] == 1:
                    break
                distance += 1
            self.sensor_distances[i] = distance
    def isOnWall(self, point):
        return self.map.map_array[int(point[0]), int(point[1])] == 1

    def isCompleted(self, point):
        return self.map.map_array[int(point[0]), int(point[1])] == 3

    def isTouching(self):
        for enemy in self.enemylist:
            if enemy.isTouching():
                return True
        return False

    def CountInSight(self):
        count = 0
        for enemy in self.enemylist:
            if enemy.is_line_of_sight_clear()[0]:
                count += 1
        return count

    def UpdateBots(self):
        for enemy in self.enemylist:
            enemy.updatePlayerPosition(self.pos)

    def steerLeft(self):
        # Similar to Java implementation, rotate the velocity vector to the left
        angle_change = 0.08
        current_angle = math.atan2(self.velocity[1], self.velocity[0])
        new_angle = current_angle + angle_change
        self.velocity = [math.cos(new_angle), math.sin(new_angle)]

    def steerRight(self):
        # Similar to Java implementation, rotate the velocity vector to the right
        angle_change = 0.08
        current_angle = math.atan2(self.velocity[1], self.velocity[0])
        new_angle = current_angle - angle_change
        self.velocity = [math.cos(new_angle), math.sin(new_angle)]

    def update(self,action):
        f = 0.1
        if (action == -1):
            self.steerLeft()
        elif (action == 1):
            self.steerRight()

        new_pos = [self.pos[0] + self.velocity[0], self.pos[1] + self.velocity[1]]
        if self.map.getdistance(new_pos) < self.map.getdistance(self.pos):
            f += 0.3
        if self.isOnWall(new_pos):
            self.crashed = True
            self.alive = False
            f -= 1
        elif self.isCompleted(new_pos):
            self.completed = True
            self.alive = False
            f += 5
        elif self.isTouching():
            print("Killed By Enemy")
            self.killed = True
            self.alive = False
            f -= 5
        else:
            self.pos = new_pos
            self.UpdateBots()
            self.count += 1
            self.update_sensor_distances()
        return f
    def isAlive(self):
        return self.alive
    def render(self, screen):
        if not self.completed:
            pygame.draw.circle(screen, (255, 0, 0), (int(self.pos[1]), int(self.pos[0])), 1)  # Draw a red dot at player position
        else:
            pygame.draw.circle(screen, (255, 255, 0), (int(self.pos[1]), int(self.pos[0])), 1)  # Draw a yellow dot at player position

    # def calcFitness(self):
    #     fitness = self.map.distanceFromStart - self.map.getdistance(self.pos)
    #     if self.completed:
    #         fitness *= 10
    #     if self.crashed:
    #         fitness /= 10
    #     if self.killed:
    #         fitness /= 15
    #     following = self.CountInSight()
    #     if following == 0 or self.killed:    
    #         self.fitness = fitness 
    #     else:
    #         self.fitness = fitness / (following * 5)
map_generator = MapGenerator(50, 50)
def eval_genomes(genomes, config):
    global gen
    gen += 1
    enemylist = map_generator.enemyList
    players = []
    nets = []
    ge = []
    population = PopulationRender(map_generator)
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        players.append(Player(enemylist,map_generator))
        ge.append(genome)

    running = True
    while running and len(players) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        i = 0
        for p in players:
            i = players.index(p)
            output = nets[i].activate(players[i].getInputs())
            a = 0
            if (output[0] < 0.5):
                a = 0
            elif (output[1] < 0.5):
                a = -1
            else:
                a = 1
            f = players[i].update(a)
            ge[i].fitness += f
            if (not players[i].isAlive()):
                ge.pop(i)
                players.pop(i)
                nets.pop(i)
            print("Population Len: ", len(players))
            population.render(players)

    pygame.quit()

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))
    

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
