# AutoDL
Automated Deep Learning framework for classifying automatically images from a folder.

## Instructions
1) Replace the images from the training and validation folders with images that you would like to classify. There are some examples of data folders with cats vs dogs and hot dog vs not hot dog. You can also add multiple labels (instead of 2). 
2) On a jupyter console or ipython, import the helper:
```
import autodl as adl

adl.run()
```

3) After running a model will be created and saved as `best_model.h5fd`. In order to visualize how your model performs, run:
```
import autodl as adl

adl.evaluate_from_url(IMG_URL) # Replace with url of image that you would like to evaluate

```

## Optional arguments
- `autodl.run()`
```
    Parameters
    ----------
    clf : estimator object
        Classifier estimator implemented using the scikit-learn interface. 

```

- `autodl.evaluate_from_url(url, model='best_model.h5)`
```
    Parameters
    ----------
    clf : estimator object
        Classifier estimator implemented using the scikit-learn interface.

```
