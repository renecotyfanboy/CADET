# *CAvity DEtection Tool* (CADET)
CADET is a machine learning pipeline trained for identification of surface brightness depressions (X-ray cavities) on noisy *Chandra* images of elliptical galaxies. The pipeline consists of a convolutional neural netwrok trained for producing pixel-wise cavity predictions, which are decomposed into individual cavities using a clustering algorithm (DBSCAN). 

The pipeline was developed as a part of my [Diploma thesis](pdfs/diploma_thesis.pdf) to improve the automation and accuracy of the detection and size-estimation of X-ray cavities. The architecture of the convolutional netwrok is inspired by [Fort et al. 2017](https://ui.adsabs.harvard.edu/abs/2017arXiv171200523F/abstract) and the used clustering algorithm is the *Sklearn* implementation of the Density-Based Spatial Clustering of Applications with Noise (DBSCAN, [Ester et al. 1996](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.121.9220)). 

![](figures/architecture.png)

---

## Requirements

`numpy`
`tensorflow`
`keras`
`sklearn`

---

## Usage



### Convolutional part

The convolutional part can be used separately to produce a pixel-wise predictions. The architecture of the convolutional network was implented using the functional *Keras* API. The architectures of individual network with trained weights were stored in the HDF5 format (compatible with *Keras*) into files *CADET_size.h5* and *CADET_search.h5*. The models can be simply loaded using the `load_model` *Keras* function.

```python
from keras.models import load_model
from keras.layers import LeakyReLU

model = load_model("CADET_size.h5", custom_objects = {"LeakyReLU": LeakyReLU})

y_pred = model.fit(X)
```

The network inputs $128x128$ images. However, to maintain the compatibility with *Keras*, the input needs to be reshaped as `X.reshape(1, 128, 128, 1)` for single image or `X.reshape(len(X), 128, 128, 1)` for multiple images.

## Example

Here we present an example of the pipeline being used on real *Chandra* images of giant elliptical galaxies.

![](figures/predictions.png)
