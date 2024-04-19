#!/usr/bin/env python3

#-*- coding: utf-8 -*-
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRSpendingByCategory(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_spending,
                   reducer=self.reducer_sum_sales),
            MRStep(reducer=self.reducer_find_max_sales_by_city)
        ]

    def mapper_get_spending(self, _, line):
        columns = line.strip().split('\t')
        date, heure, ville, categorie, somme, moyen_paiement = columns
        yield (ville, categorie), float(somme)

    def reducer_sum_sales(self, ville_categorie, somme):
        yield ville_categorie[0], (ville_categorie[1], sum(somme))

    def reducer_find_max_sales_by_city(self, ville, categorie_total):
        max_categorie, max_total = max(categorie_total, key=lambda x: x[1])
        yield ville, max_categorie

if __name__ == '__main__':
    MRSpendingByCategory.run()