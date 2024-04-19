#!/usr/bin/env python3

#-*- coding: utf-8 -*-
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRWomenClothingCashCity(MRJob):

    def steps(self):
        return [MRStep(mapper=self.mapper,reducer=self.reducer),MRStep(reducer = self.find_max)]

    def mapper(self, _, line):
        columns = line.strip().split('\t')
        date, heure, ville, categorie, somme, moyen_paiement = columns
        if categorie == "Women's Clothing" and moyen_paiement == "Cash":
            yield(ville, float(somme))
    
    def reducer(self, word, counts):
        yield(None, (word, sum(counts)))

    def find_max(self, _, word_counts):
        max_city= max(word_counts, key=lambda x: x[1])
        yield(max_city[0], max_city[1])


if __name__ == '__main__':
    MRWomenClothingCashCity.run()