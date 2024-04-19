#!/usr/bin/env python3

#-*- coding: utf-8 -*-
from mrjob.job import MRJob

class MRSumOfExpenses(MRJob):

    def mapper(self, _, line):
        columns = line.strip().split('\t')
        date, heure, ville, categorie, somme, moyen_paiement = columns
        yield(categorie, float(somme))
    
    def reducer(self, word, counts):
        yield(word, sum(counts))


if __name__ == '__main__':
    MRSumOfExpenses.run()