#!/usr/bin/env python3

#-*- coding: utf-8 -*-
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRAnagrams(MRJob):

    def mapper(self, _, line):
        words = line.lower().split()
        for word in words:
            sorted_word = ''.join(sorted(word))
            yield sorted_word, word

    def reducer(self, sorted_word, words):
        unique_words = list(words)
        if len(unique_words) > 1:
            yield sorted_word, unique_words

if __name__ == '__main__':
    MRAnagrams.run()