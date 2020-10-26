import random

import numpy as np


class Cluster:

    def __init__(self, data, k_):

        self.k = k_  # Value of K
        self.points = data  # Data
        self.centroids = []  # Centroids at the beginning of each iteration
        self.cluster = {}  # Datapoints in each cluster
        self.cluster_obj={}

    def get_centroid(self):

        self.centroids = random.choices(self.points, k=1)  # 1st data point

        while (len(self.centroids) < self.k):
            max_dist = 0
            max_pt = []
            for index_point in range(len(self.points)):
                dist = 0
                for index_k in range(len(self.centroids)):
                    dist += self.distance(self.centroids[index_k], self.points[
                        index_point])  # Calculate distance of each point from all centroids

                if (max_dist < dist):
                    max_dist = dist
                    max_pt = self.points[index_point]
            self.centroids.append(max_pt)  # Append datapoint with maximum distance

    def get_wss(self):
        error = 0
        for i in range(self.k):
            error += np.sum(np.square(self.cluster[i + 1] - self.centroids[i]))  # Objective Function
        return error

    def distance(self, x, y):
        d = np.sqrt(np.sum(np.square(x - y)))  # Calculating distance between two points
        return d

    def group(self):
        for index_point in range(len(self.points)):
            min_distance = 10000
            min_k = -1
            for index_k in range(self.k):

                dist = self.distance(self.centroids[index_k],
                                     self.points[index_point])  # Calculate distance between centroid and points

                if (min_distance > dist):
                    min_distance = dist
                    min_k = index_k
            #if (self.points[index_point] not in self.centroids):
            self.cluster[min_k + 1].append(self.points[index_point])  # Append datapoint with min distance to centroid

    def initial_cluster(self):
        for i in range(self.k):
            self.cluster[i + 1] = []
            # self.cluster[i + 1].append(np.stack(self.centroids[i]))  # Initialise clusters with centroid

    def get_mean(self):
        new_mean = []
        for i in range(1, self.k + 1):
            new_mean.append(np.sum(self.cluster[i], axis=0) / len(self.cluster[i]))  # Find mean of each cluster

        return new_mean

    def stop_check(self, new):

        if (np.sum(new - np.stack(self.centroids)) < 0.25):  # Check if mean is unchanged

            return False
        else:
            self.centroids = new
            return True

    def rewrite(self):


        for k in range(1,self.k+1):
            self.cluster_obj[k]=[]

            for f in range(len(self.cluster[k])):
                if(len(np.where(np.all(self.points==self.cluster[k][f],axis=1))[0].flatten())!=0):

                    self.cluster_obj[k].append(np.where(np.all(self.points==self.cluster[k][f],axis=1))[0].flatten()[0])

    def kmeans(self):
        self.get_centroid()  # find centroids
        iterations = 2000  # maximum iterations
        stop = True
        count = 0
        while (stop or iterations != count):
            count += 1  # iteration count
            self.initial_cluster()  # initalise cluster

            self.group()  # Cluster data points
            new_mean = self.get_mean()  # Find mean of each cluster

            stop = self.stop_check(new_mean)  # Check if mean changed

        self.rewrite()
        return self.cluster_obj
