# AutoDL
Automated Deep Learning framework for classifying automatically images from a folder.

## Instructions
1) Replace the images from the training and validation folders with images that you would like to classify. There are some examples of data folders with cats vs dogs and hot dog vs not hot dog. You can also add multiple labels (instead of 2). 
2) On a jupyter console or ipython, import the helper:
```
>>> from autodl import AutoDL
>>> ad = AutoDL('./data-cats-dogs/')
>>> ad.run()
Epoch 1/10
6/6 [==============================] - 7s 1s/step - loss: 4.9672 - acc: 0.6000 - val_loss: 0.8060 - val_acc: 0.9500

Epoch 00001: val_acc improved from -inf to 0.95000, saving model to best_model.hdf5
Epoch 2/10
6/6 [==============================] - 1s 200ms/step - loss: 1.8661 - acc: 0.8667 - val_loss: 0.8059 - val_acc: 0.9500

...
```

3) After running a model will be created and saved as `best_model.h5fd`. In order to visualize how your model performs, run:
```
>>> ad.predict_from_url(IMG_URL)
```

3.1) You can also predict from a filepath:
```
>>> ad.predict_from_url(IMG_URL)
```

Try it out on colab: https://colab.research.google.com/drive/1ATUKbyKJLSF6DMQAuIrPppGvr4MKjPoO

## Optional arguments
- `autodl.lr_finder()`: Find the optimal hyperparameter
