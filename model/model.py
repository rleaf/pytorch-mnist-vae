import torch
import torch.nn as nn

class Encoder(nn.Module):
   def __init__(self, input_size, hidden_dim, latent_size):
      super().__init__()

      self.layer_stack = nn.Sequential(
         nn.Flatten(),
         nn.Linear(input_size, hidden_dim),
         nn.ReLU(),
         nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(),
         nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(),
      )
      self.mu_layer = nn.Linear(hidden_dim, latent_size)
      self.logvar_layer = nn.Linear(hidden_dim, latent_size)

   def forward(self, x):
      z = self.layer_stack(x)
      mu = self.mu_layer(z)
      logvar = self.logvar_layer(z)
      z_tilde = reparametrize(mu, logvar)

      return z_tilde, mu, logvar


class Decoder(nn.Module):
   def __init__(self, input_size, hidden_dim, latent_size):
      super().__init__()

      self.layer_stack = nn.Sequential(
         nn.Linear(latent_size, hidden_dim),
         nn.ReLU(),
         nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(),
         nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(),
         nn.Linear(hidden_dim, input_size),
         nn.Sigmoid(),
         nn.Unflatten(dim=-1, unflattened_size=(1, 28, 28))  # (N, 1, H, W)
      )

   def forward(self, x):
      x_hat = self.layer_stack(x)

      return x_hat


class VAE(nn.Module):
   def __init__(self, input_size, latent_size=15):
      super().__init__()

      self.input_size = input_size
      self.latent_size = latent_size
      self.hidden_dim = 400
      self.encoder = Encoder(input_size, self.hidden_dim, latent_size)
      self.decoder = Decoder(input_size, self.hidden_dim, latent_size)

   def forward(self, x):
      z_tilde, mu, logvar = self.encoder(x)
      x_hat = self.decoder(z_tilde)

      return x_hat, mu, logvar


def reparametrize(mu, logvar):
   epsilon = torch.normal(0, 1, size=(mu.shape))
   sigma = torch.sqrt(torch.exp(logvar))
   z = sigma.to(mu.device) * epsilon.to(mu.device) + mu

   return z
