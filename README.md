# AA228 Final Project: Saving Artificial Intelligence Clinician

## Overview

In our work we reproduce the works of Komorowksi et al. [1] and explore methods that can address its limitations as described in  Jetter  et  al.  [2]. We describe alternatives to state representation and alternative methods of learning: Q-learning and SARSA with Value Function Approximation.

## Project folder structure

`results` - Summary analysis of policies of used algorithms

`dataset_json` - Input dataset, selected sepsis patient trajectories from MIMIC dataset in json format

`dataset_artifacts` - Different versions of processed sepsis dataset with included discrete states. Difference is in method of clustering. -vae are based on autoencoder.

`value_iterator` - implementation of model based value iterator MDP algorithm on discrete state space

`deep_sarsa` - implementation of model free SARSA algorithm with value function approximation

## References:

[1] M. Komorowski, L. A. Celi, O. Badawi, A. C. Gordon, and A. A. Faisal,“The artificial intelligence clinician learns optimal treatment strategies forsepsis  in  intensive  care,”  vol.  24,  no.  11,  pp.  1716–1720.   Number:  11Publisher: Nature Publishing Group.

[2]  R.  Jeter,  C.  Josef,  S.  Shashikumar,  and  S.  Nemati,  “Does  the  ”artificialintelligence  clinician”  learn  optimal  treatment  strategies  for  sepsis  inintensive care?,”