#!/usr/bin/env python3

#-*- coding: utf-8 -*-
from mrjob.job import MRJob

class MRSFExpenses(MRJob):

    def mapper(self, _, line):
        columns = line.strip().split('\t')
        date, heure, ville, categorie, somme, moyen_paiement = columns
        if ville == "San Francisco":
            yield(moyen_paiement, float(somme))
    
    def reducer(self, word, counts):
        yield(word, sum(counts))


if __name__ == '__main__':
    MRSFExpenses.run()