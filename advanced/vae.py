import torch
import torch.nn as nn

raw_dim = 28 * 28
hidden_dim = 128
latent_dim = 20


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # encode
        self.encoder_l1 = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.ReLU())
        # 均值
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        # 方差
        self.encoder_log_var = nn.Linear(hidden_dim, latent_dim)
        # decode
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, raw_dim),
            nn.Sigmoid())

    def encode(self, x):
        hidden = self.encoder_l1(x)
        mu = self.encoder_mu(hidden)
        log_var = self.encoder_log_var(hidden)
        return mu, log_var

    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample_z(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = self.get_eps()
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sample_z(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
