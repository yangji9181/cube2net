#!/bin/sh

./embed -entity word.txt -network network.txt -output vec.emb -binary 1 -size 100 -negative 5 -samples 1000 -iters 1 -threads 12