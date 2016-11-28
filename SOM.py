import math
import random
import psycopg2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize



class Node:

    def __init__(self, weights, x, y):
        self.weights = weights
        self.x = x
        self.y = y

    def distanceTo(self, n2):
        xleg = self.x - n2.x
        xleg *= xleg
        yleg = self.y - n2.y
        yleg *= yleg
        return xleg+yleg

    def getError(self, vector):
        error = 0

        for i in range(len(self.weights)):
            error +=  math.pow( self.weights[i] - vector[i], 2)

        return math.sqrt(error)

    def adjustWeights(self, vector, learningRate, distanceFalloff):
        for i in range(len(self.weights)):
            wt = self.weights[i]
            vt = vector[i]
            wt += distanceFalloff * learningRate * (vt - wt)
            self.weights[i] = wt




class SOM:

    START_LEARNING_RATE = 0.07
    NUM_ITERATIONS = 500
    GEOBOX = [-0.4315, 39.4196, -0.2857, 39.5045]

    def __init__(self, width, height):
        self.width = width
        self.height = height


        self.radius = max( self.width, self.height ) / 2
        self.time_constant = SOM.NUM_ITERATIONS / math.log(self.radius)


        self.nodes = [[Node([random.uniform(SOM.GEOBOX[1], SOM.GEOBOX[3]), random.uniform(SOM.GEOBOX[0], SOM.GEOBOX[2])], j, i) for i in range(width)] for j in range(height)]


    def getNeighborhoodRadius(self, iteration):
        return self.radius * math.exp(-iteration / self.time_constant)


    def getDistanceFalloff(self, distSq, radius):
        radiusSq = math.pow(radius, 2)
        return math.exp(-distSq/(2 * radiusSq))


    def getBMU(self, vector):

        min_distance = float("inf")

        for i in range(self.height):
            for j in range(self.width):
                node = self.nodes[i][j]

                distance = node.getError(vector)
                if distance < min_distance:
                    min_distance = distance
                    bmu = node

        return bmu


    def train(self, train_set):
        iteration = 0
        learning_rate = SOM.START_LEARNING_RATE

        while( iteration < SOM.NUM_ITERATIONS ):
            nbhRadius = self.getNeighborhoodRadius(iteration)


            for t in range(len(train_set)):
                tweet = train_set[t]

                bmu = self.getBMU(tweet)

                for i in range(self.height):
                    for j in range(self.width):
                        temp = self.nodes[i][j]
                        distance = bmu.distanceTo(temp)

                        if distance <= nbhRadius * nbhRadius:
                            dFalloff = self.getDistanceFalloff(distance, nbhRadius)
                            temp.adjustWeights(tweet, learning_rate, dFalloff)
            '''
            print("Iteration " + str(iteration))
            for i in range(self.height):
                for j in range(self.width):
                    print(self.nodes[i][j].weights)

            print()
            '''

            iteration += 1
            learning_rate = SOM.START_LEARNING_RATE * math.exp( -iteration / SOM.NUM_ITERATIONS)

            return self.nodes




    def classify(self, data):

        hist = [[[] for j in range(self.width)] for i in range(self.height)]
        stddev = [[0] * self.width for i in range(self.height)]

        for tweet in data:
            bmu = self.getBMU(tweet)
            hist[bmu.x][bmu.y] += [tweet]
            stddev[bmu.x][bmu.y] += math.pow(tweet[0] - bmu.weights[0], 2) + math.pow(tweet[1] - bmu.weights[1], 2)



        for i in range(self.height):
            for j in range(self.width):
                #print(hist[i][j])
                #stddev[i][j] = np.std(hist[i][j])
                stddev[i][j] /= len(hist[i][j])
                stddev[i][j] = math.sqrt(stddev[i][j])
                #print(stddev[i][j])



        return self.DBindex(stddev)


    def DBindex(self, stddev):

        db = 0.
        for i in range(self.height):
            for j in range(self.width):
                maxValue = float('infinity')
                maxValue = - maxValue


                for i2 in range(self.height):
                    for j2 in range(self.width):
                        value = 0.
                        if i != i2 and j != j2:
                            value += stddev[i][j]
                            #print(value)
                            value += stddev[i2][j2]
                            #print(value)
                            #print(self.nodes[i][j].getError(self.nodes[i2][j2].weights))
                            value /= (self.nodes[i][j].getError(self.nodes[i2][j2].weights))


                            if value > maxValue:
                                maxValue = value


                db += maxValue

        db /= (self.width*self.height)

        return db







def main():
    conn = psycopg2.connect("dbname=postgis_22_sample user=postgres password=postgres")

    cur = conn.cursor()
    query = "SELECT ST_Y(twe_coordenadas), ST_X(twe_coordenadas) FROM tweet"
    cur.execute(query)

    result = cur.fetchall()

    cur.close()
    conn.close()

    data = [list(elem) for elem in result]



    for i in range(3, 20):
        for j in range(3, 20):
            avg = 0.
            min = float("infinity")
            max = -min
            #for r in range(10):
            if i * j <= 100:
                som = SOM(i,j)
                som.train(data)
                db = som.classify(data)
                if db < min:
                    min = db
                if db > max:
                    max = db
                avg += db
                print(db)


            avg /= 10
            print("DB index for size {} x {}: {} (min) {} (max) {} (avg) ".format(i,j,min, max, avg))


    '''
    som = SOM(3, 5)
    nodes = som.train(data)

    f = open('nodos.csv', 'w', encoding='utf8')
    for i in range(5):
        for j in range(3):
            mystr = "{} , {}\n".format(nodes[i][j].weights[0], nodes[i][j].weights[1])
            f.write(mystr)

    '''

main()
