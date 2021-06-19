# *CAvity DEtection Tool* (CADET)
CADET is a machine learning pipeline trained for identification of surface brightness depressions (X-ray cavities) on noisy *Chandra* images of elliptical galaxies. The pipeline consists of a convolutional neural netwrok trained for producing pixel-wise cavity predictions, which are afterwards decomposed into individual cavities using a clustering algorithm (DBSCAN).\ref{a}

The pipeline was developed as a part of my [Diploma thesis](pdfs/diploma_thesis.pdf) (not defended yet) to improve the automation and accuracy of the detection and size-estimation of X-ray cavities. The architecture of the convolutional netwrok consists of 5 convolutional block, each resembling an inception layer, and it was inspired by [Fort et al. 2017](https://ui.adsabs.harvard.edu/abs/2017arXiv171200523F/abstract). The utilized clustering algorithm is the *Sklearn* implementation of the Density-Based Spatial Clustering of Applications with Noise (DBSCAN, [Ester et al. 1996](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.121.9220)).

![obraz\lable{a}](figures/architecture.png)

The convolutional neural network was trained on artificial images which were produced from generated 3D beta models ([Cavaliere et al. 1978](https://ui.adsabs.harvard.edu/abs/1978A%26A....70..677C/abstract)) into which we randomly inserted ellipsoidal cavities. The parameter ranges and distributions used for generating the galactic models and X-ray cavities were estimated from measurements (for more info see the [Diploma thesis](pdfs/diploma_thesis.pdf)). The produced models were summed into 2D images a noised using Poisson statistics to resemble real *Chandra*-like images.

---

## Requirements

`astropy`\
`numpy`\
`keras`\
`sklearn`\
`tensorflow`

---

## Usage

Both the *CADET_search* and *CADET_size* pipelines are composed as selfstanding scripts. Discrimination threshold for the *CADET_search* pipeline was set 0.9 to supress false positive detections, while the threshold of the *CADET_size* pipeline was set to 0.55 so the predicted volumes are not underestimated nor overestimated (for more info see the [Diploma thesis](pdfs/diploma_thesis.pdf); not defended yet). However, the thresholds of both pipelines are changeable and can be set to an arbitrary value between 0 and 1.


### Convolutional part

The convolutional part can be used separately to produce pixel-wise predictions. Since the architecture of the convolutional network was implemented using the functional *Keras* API, the architectures together with trained weights could have been stored in the HDF5 format (*CADET_size.h5*, *CADET_search.h5*). The trained models can be simply loaded using the `load_model` *Keras* function.

```python
from keras.models import load_model
from keras.layers import LeakyReLU

model = load_model("CADET_size.h5", custom_objects = {"LeakyReLU": LeakyReLU})

y_pred = model.predict(X)
```

The network inputs 128x128 images. However, to maintain the compatibility with *Keras*, the input needs to be reshaped as `X.reshape(1, 128, 128, 1)` for single image or as `X.reshape(len(X), 128, 128, 1)` for multiple images.

## Example

Here we present an example of the pipeline being used on real *Chandra* images of giant elliptical galaxies.

![](figures/predictions.png)
