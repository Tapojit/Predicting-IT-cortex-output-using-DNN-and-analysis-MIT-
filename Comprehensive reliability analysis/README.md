# Determining reliability of IT (Inferior Temporal) neural outputs predicted using a given covariate matrix.

This is a walkthrough of the code implementations which can be used to determine reliability of predicted IT(Inferior Temporal) neural outputs. Description of the terminologies are available in Readme.md file located in the previous directory.

Reliability is measured in terms of explained variance(%) between predicted IT matrix vectors and actual IT matrix vectors. The closer the explained variance is to 100%, the more similar/reliable the predicted IT matrix vectors are. It is calculated using this formula, where the predicted vector is **f** and the actual vector is **y** :
