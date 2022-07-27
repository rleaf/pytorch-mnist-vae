# pytorch-mnist-vae

A simple Pytorch implementation of a Variational Autoencoder trained on the MNIST dataset.


1) **Train the model.** `train.py` will download MNIST and train the model. Once the model is trained, the weights will be saved. `train.py -h` to see configuration.
<br>
2) **Generation** `generate.py` will output *'vae_generation.jpg'*. `generate.py -h` to see configuration.

All files downloaded, saved, and generated are default saved in current directory `./`.