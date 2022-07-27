# pytorch-mnist-vae

A simple Pytorch implementation of a Variational Autoencoder trained on the MNIST dataset.


1) **Train the model.** `train.py` will download MNIST and train the model. Once the model is trained, the weights will be saved. `train.py -h` to see configuration.

2) **Generation** `generate.py` will output *'vae_generation.jpg'*. `generate.py -h` to see configuration.

All files downloaded, saved, and generated are default saved in current directory `./`.



![vae_generation](https://user-images.githubusercontent.com/1999954/181287360-4213e328-7e83-4b79-8ab0-1320ee159c5e.jpg)
