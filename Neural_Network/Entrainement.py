# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:27:04 2022
@author: maxime.mines-ales
"""

from Reseau import Reseau;
from keras.datasets import mnist
import numpy as np
import random as rd

# config
image_size = 28  # hauteur, largeur
image_pixels = image_size * image_size

# initialisation de la banque de donnnées
(train_X, train_y), (test_X, test_y) = mnist.load_data()  # test = 10_000; train = 60_000

# conversion des images NB 255 en NB 00.1-1 en multipliant par fac
fac = 0.99 / 255

train_imgs = np.asfarray(train_X[:, 0:]) * fac + 0.01  # 60000 images le +0.01 pour éviter la valeur 0
test_imgs = np.asfarray(test_X[:, 0:]) * fac + 0.01  # 10000 images


def matrice_en_vecteur(matrice):
    return np.array(matrice).ravel()


matrice_en_vecteur(test_imgs[0])

nombre = 10


def one_hot_representation(liste_vecteurs):
    L = np.array([[0.01 for k in range(nombre)]] * len(liste_vecteurs))
    for k, val in enumerate(liste_vecteurs):
        L[k][val] = 0.99
    return L


test_one_hot_representation = one_hot_representation(test_y)
train_one_hot_representation = one_hot_representation(train_y)

# réseau de neurones
reseau = Reseau(image_pixels, 20, 15, 12, 10, 0.002)

# entrainement du réseau sur la base de données d'entrainement
rd.shuffle(train_X)
for i in range(len(train_X)):
    reseau.entrainer(np.append(matrice_en_vecteur(train_imgs[i]), 0), train_one_hot_representation[i])

# performance
perf = 0

for i in range(len(test_X)):
    k = reseau.question(np.append(matrice_en_vecteur(test_imgs[i]), 0))
    if k == test_y[i]:
        perf += 1

print(str(perf / len(test_X) * 100) + " % de réussite")
