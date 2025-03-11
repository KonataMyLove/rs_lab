from collections import defaultdict
from tqdm import tqdm

import random
import math
import pdb

class ItemCF(object):
    def __init__(self):
        self.threshold = 0.75
        self.n_sim_movies = 20
        self.n_rec_movies = 10

        self.trainSet = defaultdict(dict)
        self.testSet = defaultdict(dict)

        self.movie_sim_matrix = defaultdict(dict)
        self.movie_popular = defaultdict(int)

        self.train_len = 0
        self.test_len = 0

    def get_dataset(self, file_path):
        for line in self.read_file(file_path):
            num = random.random()
            user, movie, rating, timestamp = line.split(',')
            if num <= self.threshold:
                self.trainSet[user][movie] = rating
                self.train_len += 1
            else:
                self.testSet[user][movie] = rating
                self.test_len += 1
        print('Training Set length: %d' % self.train_len)
        print('Testing Set length: %d' % self.test_len)
        return

    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')

    def compute_sim(self):
        print('Strat computing item similarity matrix')
        for user, movies in self.trainSet.items():
            for movie in movies:
                self.movie_popular[movie] += 1

        for user, movies in tqdm(self.trainSet.items()):
            for u in movies:
                for v in movies:
                    if u == v:
                        continue
                    if v not in self.movie_sim_matrix[u]:
                        self.movie_sim_matrix[u][v] = 0
                    self.movie_sim_matrix[u][v] += 1

        for m1, sim_list in self.movie_sim_matrix.items():
            if self.movie_popular[m1] == 0:
                continue
            for m2, numerator in sim_list.items():
                if self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                    continue    
                denominator = math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
                self.movie_sim_matrix[m1][m2] = numerator / denominator
        print('Item similarity matrix complete')
        return

    def rec_serv(self, user_id):
        # print('running recommend service')
        rec = defaultdict(int)
        watched_movies = self.trainSet[user_id]
        for movie, rating in watched_movies.items():
            for movie, w in sorted(self.movie_sim_matrix[movie].items(), key=lambda x:x[1], reverse=True)[:self.n_sim_movies]:
                if movie in watched_movies:
                    continue
                rec[movie] += w * float(rating)

        return sorted(rec.items(), key=lambda x:x[1], reverse=True)[:self.n_rec_movies]
    
    def evaluate(self):
        print('start evaluating')
        hit = 0
        total_recommend_movies = 0  # 查全率，recall
        total_watched_movies = 0    # 查准率，precision

        for i, user_id in tqdm(enumerate(self.trainSet)):
            test_movies = self.testSet.get(user_id, {})
            rec_list = self.rec_serv(user_id)
            for rec_movie in rec_list:
                if rec_movie[0] in test_movies:
                    hit += 1
            total_recommend_movies += self.n_rec_movies
            total_watched_movies += len(test_movies)

        recall = hit / (1.0 * total_recommend_movies)
        precision = hit / (1.0 * total_watched_movies)
        print('hit:%d' % hit)
        return (recall, precision)
        

if __name__ == '__main__':
    fp = '/home/zhuliqing/code/AI-RecommenderSystem/Recall/CollaborativeFiltering/ml-latest-small/ratings.csv'
    cf = ItemCF()
    cf.get_dataset(fp)
    cf.compute_sim()
    # print(cf.rec_serv('1'))
    R, P = cf.evaluate()
    print('Recall: %f, Precision: %f' % (R, P))

