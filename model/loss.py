import torch
import torch.nn as nn

def loss(x_hat, x, mu, logvar):
   """
   KL divergence simplifies to what is shown below because
   the encoder distribution: $q_phi(z|x) sim mathcal{N}(mu, diag(Sigma))$ AND
   the prior distribution: $p_theta(z) sim mathcal{N}(0, I)$

   eq (2.31) @ 
   https://ryli.design/blog/tractability-and-optimization-of-vae#bayes
   """

   reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
   kl_divergence = -0.5 * (torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar)))
   loss = reconstruction_loss + kl_divergence
   loss /= mu.shape[0]

   return loss
