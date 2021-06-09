# *CAvity DEtection Tool* (CADET)
Convolutional pipeline trained for detection and size estimation of brightness depressions (X-ray cavities) on noisy *Chandra* images of elliptical galaxies. The pipeline consists of convolutional neural network composed from inception blocks and a *Sklearn* implementation of the Density-based spatial clustering of applications with noise (DBSCAN, [Ester et al. 1996](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.121.9220). The convolutional network is trained for producing pixel-wise cavity prediction, which is decomposed into individual cavities using the DBSCAN algorithm.

The pipeline was developed as a part of my [Diploma thesis](documents/diploma_thesis.pdf) to improve the automation and accuracy in the detection and size-estimation process of X-ray cavities. The architecture is inspired by [Fort et al. 2017](https://ui.adsabs.harvard.edu/abs/2017arXiv171200523F/abstract)

<img src="figures/architecture.png">
