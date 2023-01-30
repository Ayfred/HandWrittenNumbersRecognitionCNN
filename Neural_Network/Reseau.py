# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:51:14 2022

@author: maxime.mines-ales
"""

import numpy as np
import scipy.special
from  termcolor import colored


class Reseau:

    def __init__(self, entree, intermediaire1, intermediaire2, intermediaire3, sortie, apprentissage):
        self.entree = entree
        self.intermediaire1 = intermediaire1
        self.intermediaire2 = intermediaire2
        self.intermediaire3 = intermediaire3
        self.sortie = sortie

        # UP : On a initialisé le réseau, et on stocke les nombres de neurones par couches (entree = 784 et sortie = 10)

        self.liaison_entree_intermediaire1 = (np.random.rand(self.intermediaire1, self.entree) - 0.5)
        np.c_[self.liaison_entree_intermediaire1, np.ones(len(self.liaison_entree_intermediaire1))]
        np.r_[self.liaison_entree_intermediaire1, np.ones(len(self.liaison_entree_intermediaire1))]
        # print("dsfgshed : " + str(self.liaison_entree_intermediaire1.shape))
        self.liaison_intermediaire1_intermediaire2 = (np.random.rand(self.intermediaire2, self.intermediaire1) - 0.5)
        np.c_[self.liaison_entree_intermediaire2, np.ones(len(self.liaison_entree_intermediaire2))]
        np.r_[self.liaison_entree_intermediaire2, np.ones(len(self.liaison_entree_intermediaire2))]
        self.liaison_intermediaire2_intermediaire3 = (np.random.rand(self.intermediaire3, self.intermediaire2) - 0.5)
        np.c_[self.liaison_entree_intermediaire3, np.ones(len(self.liaison_entree_intermediaire3))]
        np.r_[self.liaison_entree_intermediaire3, np.ones(len(self.liaison_entree_intermediaire3))]
        self.liaison_intermediaire3_sortie = (np.random.rand(self.sortie, self.intermediaire3) - 0.5)
        np.c_[self.liaison_intermediaire3_sortie, np.ones(len(self.liaison_intermediaire3_sortie))]
        np.r_[self.liaison_intermediaire3_sortie, np.ones(len(self.liaison_intermediaire3_sortie))]

        # UP : On stocke les matrices des poids m1, m2, m3 et m4
        # NB : On ajoute une ligne pour pour faire apparraître les bj (biais) sur la dernière ligne

        self.app = apprentissage

        # UP : En principe, alfa vaut 10^-3 le plus souvent

        # fonction d'activation sigmoide
        self.fonction_activation = lambda x: scipy.special.expit(x)

    def entrainer(self, liste_image, liste_valeur):

        entree = np.array(liste_image, ndmin=2).T
        valeur = np.array(liste_valeur, ndmin=2).T

        intermediaire1_entree = np.dot(self.liaison_entree_intermediaire1, entree);
        intermediaire1_s = self.fonction_activation(intermediaire1_entree)


        intermediaire2_intermediaire1 = np.dot(self.liaison_intermediaire1_intermediaire2, intermediaire1_s);
        intermediaire2_s = self.fonction_activation(intermediaire2_intermediaire1)
        np.append(intermediaire2_s, 1)

        intermediaire3_intermediaire2 = np.dot(self.liaison_intermediaire2_intermediaire3, intermediaire2_s);
        intermediaire3_s = self.fonction_activation(intermediaire3_intermediaire2)
        np.append(intermediaire3_s, 1)

        sortie_intermediaire3 = np.dot(self.liaison_intermediaire3_sortie, intermediaire3_s);
        sortie_s = self.fonction_activation(sortie_intermediaire3)

        erreur_sortie = valeur - sortie_s
        erreur_intermediaire3 = np.dot(self.liaison_intermediaire3_sortie.T, erreur_sortie)
        erreur_intermediaire2 = np.dot(self.liaison_intermediaire2_intermediaire3.T, erreur_intermediaire3)
        erreur_intermediaire1 = np.dot(self.liaison_intermediaire1_intermediaire2.T, erreur_intermediaire2)

        self.liaison_intermediaire3_sortie += self.app * np.dot(erreur_sortie * sortie_s * (1.0 - sortie_s),
                                                                np.transpose(intermediaire3_s))
        self.liaison_intermediaire2_intermediaire3 += self.app * np.dot(
            erreur_intermediaire3 * intermediaire3_s * (1.0 - intermediaire3_s), np.transpose(intermediaire2_s))
        self.liaison_intermediaire1_intermediaire2 += self.app * np.dot(
            erreur_intermediaire2 * intermediaire2_s * (1.0 - intermediaire2_s), np.transpose(intermediaire1_s))
        self.liaison_entree_intermediaire1 += self.app * np.dot(
            erreur_intermediaire1 * intermediaire1_s * (1.0 - intermediaire1_s), np.transpose(entree))

    def question(self, liste_entree):

        entree = np.array(liste_entree, ndmin=2).T
        intermediaire1_entree = np.dot(self.liaison_entree_intermediaire1, entree)
        intermediaire1_s = self.fonction_activation(intermediaire1_entree)

        intermediaire2_intermediaire1 = np.dot(self.liaison_intermediaire1_intermediaire2, intermediaire1_s)
        intermediaire2_s = self.fonction_activation(intermediaire2_intermediaire1)

        intermediaire3_intermediaire2 = np.dot(self.liaison_intermediaire2_intermediaire3, intermediaire2_s)
        intermediaire3_s = self.fonction_activation(intermediaire3_intermediaire2)

        sortie_intermediaire3 = np.dot(self.liaison_intermediaire3_sortie, intermediaire3_s)
        sortie_s = self.fonction_activation(sortie_intermediaire3)

        x, i = 0, 0
        for k, val in enumerate(sortie_s):
            if val > x:
                i, x = k, val

        return i


