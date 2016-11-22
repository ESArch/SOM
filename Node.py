import math
import random
import psycopg2

import matplotlib.pyplot as plt
import matplotlib.cm

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize



class Node:

    def __init__(self, weights=[0, 0]):
        self.weights = weights


    def getDistance(self, vector):
        distance = 0

        for i in range(len(self.weights)):
            distance +=  math.pow( self.weights[i] - vector[i], 2)

        return math.sqrt(distance)

    def adjustWeights(self, vector, learningRate, distanceFalloff):
        for i in range(len(self.weights)):
            wt = self.weights[i]
            vt = vector[i]
            wt += distanceFalloff * learningRate * (vt - wt)
            self.weights[i] = wt




class SOM:

    START_LEARNING_RATE = 0.00001
    NUM_ITERATIONS = 50
    GEOBOX = [-0.4315, 39.4196, -0.2857, 39.5045]

    def __init__(self, nNodes):
        self.nNodes = nNodes


        self.radius = max(SOM.GEOBOX[3]- SOM.GEOBOX[1], SOM.GEOBOX[2]- SOM.GEOBOX[0] ) / 2
        self.time_constant = SOM.NUM_ITERATIONS / math.log(self.radius)


        self.nodes = []
        for i in range(nNodes):
            lat = random.uniform(SOM.GEOBOX[1], SOM.GEOBOX[3])
            lon = random.uniform(SOM.GEOBOX[0], SOM.GEOBOX[2])

            node = Node([lat,lon])

            self.nodes.append(node)

        for node in self.nodes:
            print(node.weights)


    def getNeighborhoodRadius(self, iteration):
        return self.radius * math.exp(-iteration / self.time_constant)


    def getDistanceFalloff(self, distSq, radius):
        radiusSq = math.pow(radius, 2)
        return math.exp(-distSq/(2 * radiusSq))


    def getBMU(self, vector):

        min_distance = float("inf")

        for i in range(self.nNodes):
            node = self.nodes[i]

            distance = node.getDistance(vector)
            if distance < min_distance:
                min_distance = distance
                bmu = node

        return bmu


    def train(self, train_set):
        iteration = 0
        learning_rate = SOM.START_LEARNING_RATE

        while( iteration < SOM.NUM_ITERATIONS ):
            nbhRadius = self.getNeighborhoodRadius(iteration)

            for i in range(len(train_set)):
                tweet = train_set[i]

                bmu = self.getBMU(tweet)

                for j in range(self.nNodes):

                    distance = self.nodes[j].getDistance(tweet)

                    if  distance <= nbhRadius:
                        dFalloff = self.getDistanceFalloff(distance, nbhRadius)
                        self.nodes[j].adjustWeights(tweet, learning_rate, dFalloff)

            iteration += 1
            learning_rate = SOM.START_LEARNING_RATE * math.exp( -iteration / SOM.NUM_ITERATIONS)


            print('\nIteracion: ', iteration)
            for node in self.nodes:
                print(node.weights)

        f = open('nodos.csv','w')
        for node in self.nodes:
            mystr = str(node.weights[0]) + ',' + str(node.weights[1])
            f.write(mystr+'\n')




    def plot_point(self, lat, lon):
        print()







def main():
    conn = psycopg2.connect("dbname=postgis_22_sample user=postgres password=postgres")

    cur = conn.cursor()
    query = "SELECT ST_Y(twe_coordenadas), ST_X(twe_coordenadas) FROM tweet LIMIT 10000"
    cur.execute(query)

    result = cur.fetchall()

    cur.close()
    conn.close()

    data = [list(elem) for elem in result]

    som = SOM(20)

    som.train(data)

main()
