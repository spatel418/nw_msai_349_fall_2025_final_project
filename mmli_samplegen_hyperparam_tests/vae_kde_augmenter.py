"""
VAE-KDE Data Augmentation Module
================================
Variational Autoencoder with Kernel Density Estimation for molecular data augmentation.
Based on the approach used for Vaskas dataset.
"""

import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
import time
import warnings
warnings.filterwarnings("ignore")


class MolecularVAE(nn.Module):
    """
    Variational Autoencoder for molecular descriptors with 2-layer architecture
    """
    def __init__(self, input_dim, latent_dim=16, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.1):
        super(MolecularVAE, self).__init__()
        
        self.encoder_layer1 = nn.Linear(input_dim, hidden_dim1)
        self.encoder_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim2, latent_dim)
        
        self.decoder_layer1 = nn.Linear(latent_dim, hidden_dim2)
        self.decoder_layer2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.decoder_output = nn.Linear(hidden_dim1, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.encoder_layer1(x))
        h1 = self.dropout(h1)
        h2 = F.relu(self.encoder_layer2(h1))
        mu = self.fc_mu(h2)
        log_var = self.fc_var(h2)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        h1 = F.relu(self.decoder_layer1(z))
        h2 = F.relu(self.decoder_layer2(h1))
        reconstructed = self.decoder_output(h2)
        return reconstructed
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var


class VAEKDEAugmenter:
    """
    Implementation using VAE, KDE, and nearest neighbors for molecular data augmentation.
    """
    def __init__(
        self,
        latent_dim=16,
        hidden_dims=[64, 32],
        num_regions=10,
        balance_strategy='equal',
        min_samples_per_region=None,
        beta=0.1,
        noise_factor=0.05,
        learning_rate=0.001,
        batch_size=32,
        epochs=200,
        early_stopping=20,
        n_neighbors=5,
        dropout_rate=0.1,
        use_latent_neighbors=True,
        balance_factor=0.3,
        device=None,
        random_state=13,
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_regions = num_regions
        self.balance_strategy = balance_strategy
        self.min_samples_per_region = min_samples_per_region
        self.beta = beta
        self.noise_factor = noise_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.n_neighbors = n_neighbors
        self.dropout_rate = dropout_rate
        self.use_latent_neighbors = use_latent_neighbors
        self.balance_factor = balance_factor
        
        if device is None:
            # Check for GPU availability: CUDA (NVIDIA) or MPS (Apple M1/M2)
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.vae = None
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def fit(self, X, y, region_edges=None, region_assignments=None):
        print(f"Using device: {self.device}")
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
        
        X_values = np.nan_to_num(X_values, nan=0.0)
        X_scaled = self.scaler.fit_transform(X_values)
        
        input_dim = X_scaled.shape[1]
        self.vae = MolecularVAE(
            input_dim=input_dim, 
            latent_dim=self.latent_dim, 
            hidden_dim1=self.hidden_dims[0],
            hidden_dim2=self.hidden_dims[1],
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        self._train_vae(X_scaled)
        
        z_mean, z_log_var = self._encode_data(X_scaled)
        
        n_samples = len(X_scaled)
        n_neighbors = min(self.n_neighbors, n_samples-1)
        print(f"Using {n_neighbors} nearest neighbors for {n_samples} samples")
        
        if self.use_latent_neighbors:
            print("Finding neighbors in VAE latent space")
            neighbor_data = z_mean
        else:
            print("Finding neighbors in original feature space")
            neighbor_data = X_scaled
            
        self.nn = NearestNeighbors(
            n_neighbors=n_neighbors+1,
            algorithm='auto'
        ).fit(neighbor_data)
        
        self.distances, self.indices = self.nn.kneighbors()
        
        if region_edges is not None and region_assignments is not None:
            self.region_edges = region_edges
            self.region_assignments = region_assignments
            print("Using pre-existing region information from stratified split")
        else:
            self.region_edges = np.percentile(y_values, np.linspace(0, 100, self.num_regions + 1))
            if len(np.unique(self.region_edges)) < len(self.region_edges):
                self.region_edges = np.linspace(np.min(y_values), np.max(y_values), self.num_regions + 1)
            self.region_assignments = np.digitize(y_values, self.region_edges[1:-1])
            print("Created new region information for augmentation")
        
        self.region_counts = np.bincount(self.region_assignments, minlength=self.num_regions)
        
        if self.balance_strategy == 'equal':
            self.target_counts = np.full_like(self.region_counts, np.max(self.region_counts))
        elif self.balance_strategy == 'proportional':
            max_count = np.max(self.region_counts)
            self.target_counts = np.maximum(self.region_counts, int(max_count * 0.5))
        elif self.balance_strategy == 'increase':
            if self.min_samples_per_region is None:
                raise ValueError("min_samples_per_region must be specified when using 'increase' strategy")
            max_natural_count = np.max(self.region_counts)
            effective_target = max(self.min_samples_per_region, max_natural_count)
            self.target_counts = np.full_like(self.region_counts, effective_target)
        else:
            self.target_counts = np.full_like(self.region_counts, np.max(self.region_counts))
        
        self.samples_to_generate = np.maximum(0, self.target_counts - self.region_counts)
        
        print("Region distribution before augmentation:")
        for i in range(self.num_regions):
            region_range = f"{self.region_edges[i]:.2f} to {self.region_edges[i+1]:.2f}"
            print(f"  Region {i+1} ({region_range}): {self.region_counts[i]} samples -> Target: {self.target_counts[i]}")
        
        self.X_scaled = X_scaled
        self.X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.y = y_values
        
        y_reshaped = y_values.reshape(-1, 1)
        bandwidth, log_likelihood = self._check_kde_quality(y_values)
        self.kde = KernelDensity(bandwidth=float(bandwidth)).fit(y_reshaped)
        self.density = np.exp(self.kde.score_samples(y_reshaped))
        
        return self
    
    def _check_kde_quality(self, y_values, max_iterations=3):
        y_reshaped = y_values.reshape(-1, 1)
        
        bandwidth = 0.5 * y_values.std() * (len(y_values) ** -0.2)
        if len(y_values) < 30:
            bandwidth *= 1.5
        if bandwidth <= 0:
            bandwidth = 1.0
        
        kde = KernelDensity(bandwidth=bandwidth).fit(y_reshaped)
        log_likelihood = np.mean(kde.score_samples(y_reshaped))
        
        print(f"Initial KDE quality: bandwidth={bandwidth:.4f}, log-likelihood={log_likelihood:.4f}")
        
        if log_likelihood >= -3.0:
            return bandwidth, log_likelihood
        
        bandwidth_increase = bandwidth * 1.5
        kde_increase = KernelDensity(bandwidth=bandwidth_increase).fit(y_reshaped)
        ll_increase = np.mean(kde_increase.score_samples(y_reshaped))
        
        bandwidth_decrease = max(bandwidth * 0.75, 0.01)
        kde_decrease = KernelDensity(bandwidth=bandwidth_decrease).fit(y_reshaped)
        ll_decrease = np.mean(kde_decrease.score_samples(y_reshaped))
        
        print(f"Increased bandwidth: {bandwidth_increase:.4f}, log-likelihood: {ll_increase:.4f}")
        print(f"Decreased bandwidth: {bandwidth_decrease:.4f}, log-likelihood: {ll_decrease:.4f}")
        
        options = [
            (bandwidth, log_likelihood),
            (bandwidth_increase, ll_increase),
            (bandwidth_decrease, ll_decrease)
        ]
        
        bandwidth, log_likelihood = max(options, key=lambda x: x[1])
        print(f"Selected bandwidth: {bandwidth:.4f}, log-likelihood: {log_likelihood:.4f}")
        
        if log_likelihood < -3.0 and max_iterations > 1:
            return self._check_kde_quality(y_values, max_iterations=max_iterations-1)
        
        return bandwidth, log_likelihood
    
    def _train_vae(self, X_scaled):
        print(f"Training VAE with {self.latent_dim} latent dimensions")
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=min(self.batch_size, len(X_scaled)), 
            shuffle=True
        )
        
        optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.vae.train()
            epoch_loss = 0
            
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                recon_x, mu, log_var = self.vae(batch_x)
                
                recon_loss = F.mse_loss(recon_x, batch_x, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + self.beta * kl_loss
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.6f}")
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def _encode_data(self, X_scaled):
        self.vae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            mu, log_var = self.vae.encode(X_tensor)
        return mu.cpu().numpy(), log_var.cpu().numpy()
    
    def generate_samples(self):
        print("Generating synthetic samples with KDE-based balanced upsampling...")
        start_time = time.time()
        
        X_synthetic_list = []
        y_synthetic_list = []
        
        self.vae.eval()
        generated_per_region = np.zeros(self.num_regions, dtype=int)
        
        z_mean, z_log_var = self._encode_data(self.X_scaled)
        
        for region_idx in range(self.num_regions):
            if self.samples_to_generate[region_idx] <= 0:
                continue
                
            region_min = self.region_edges[region_idx]
            region_max = self.region_edges[region_idx + 1]
            region_center = (region_min + region_max) / 2
            region_width = max(region_max - region_min, 1e-10)
            
            n_synthetic = self.samples_to_generate[region_idx]
            print(f"Region {region_idx+1}: Generating {n_synthetic} synthetic samples")
            
            distances_to_center = np.abs(self.y - region_center) / region_width
            region_probs = np.exp(-distances_to_center * 3)
            
            inverse_density = 1.0 / (self.density + 1e-10)
            inverse_density_probs = inverse_density / inverse_density.sum()
            uniform_probs = np.ones_like(self.density) / len(self.density)
            
            base_probs = (1 - self.balance_factor) * inverse_density_probs + self.balance_factor * uniform_probs
            combined_probs = region_probs * base_probs
            combined_probs /= combined_probs.sum()
            
            for _ in range(n_synthetic):
                idx = np.random.choice(len(self.X_scaled), p=combined_probs)
                
                neighbor_indices = self.indices[idx, 1:]
                neighbor_idx = np.random.choice(neighbor_indices)
                
                mu1 = z_mean[idx]
                mu2 = z_mean[neighbor_idx]
                
                lambd = np.random.uniform(0.2, 0.8)
                z_new = lambd * mu1 + (1 - lambd) * mu2
                z_new += np.random.normal(0, self.noise_factor, size=z_new.shape)
                
                with torch.no_grad():
                    z_new_tensor = torch.FloatTensor(z_new.reshape(1, -1)).to(self.device)
                    X_synthetic = self.vae.decode(z_new_tensor).cpu().numpy()[0]
                
                X_synthetic_list.append(X_synthetic)
                
                y_synthetic = lambd * self.y[idx] + (1 - lambd) * self.y[neighbor_idx]
                y_synthetic_list.append(y_synthetic)
                
                synthetic_region = np.digitize([y_synthetic], self.region_edges[1:-1])[0]
                generated_per_region[synthetic_region] += 1
        
        if len(X_synthetic_list) == 0:
            return np.array([]).reshape(0, self.X_scaled.shape[1]), np.array([])
        
        X_synthetic_array = np.array(X_synthetic_list)
        y_synthetic_array = np.array(y_synthetic_list)
        
        X_synthetic_original = self.scaler.inverse_transform(X_synthetic_array)
        
        y_combined = np.concatenate([self.y, y_synthetic_array])
        y_regions_combined = np.digitize(y_combined, self.region_edges[1:-1])
        region_counts_combined = np.bincount(y_regions_combined, minlength=self.num_regions)
        
        print("\nRegion distribution after augmentation:")
        for i in range(self.num_regions):
            region_range = f"{self.region_edges[i]:.2f} to {self.region_edges[i+1]:.2f}"
            original = self.region_counts[i]
            added = generated_per_region[i]
            new_total = region_counts_combined[i]
            print(f"  Region {i+1} ({region_range}): {original} + {added} = {new_total} samples")
        
        end_time = time.time()
        print(f"Generated {len(X_synthetic_list)} synthetic samples in {end_time - start_time:.2f} seconds")
        
        return X_synthetic_original, y_synthetic_array
    
    def augment_dataset(self, X, y, region_edges=None, region_assignments=None):
        self.fit(X, y, region_edges, region_assignments)
        X_synthetic, y_synthetic = self.generate_samples()
        
        if len(X_synthetic) == 0:
            return X.copy(), y.copy() if isinstance(y, pd.Series) else y
        
        X = X.copy()
        X["target_region"] = region_assignments
        
        X_synth_df = pd.DataFrame(X_synthetic, columns=X.columns.drop(['target_region']))
        X_synth_df['target_region'] = np.digitize(y_synthetic, self.region_edges[1:-1])
        
        X_augmented = pd.concat([X, X_synth_df], ignore_index=True)
        
        if isinstance(y, pd.Series):
            y_augmented = pd.concat([y.reset_index(drop=True), pd.Series(y_synthetic, name=y.name)], ignore_index=True)
        else:
            y_augmented = np.concatenate([y, y_synthetic])
            
        return X_augmented, y_augmented


def data_augment_vae_kde(df_train, target_column, bin_edges, num_bins=10, 
                         balance_strategy='equal', augmented_bucket_size=None, 
                         region_assignments=None, save_dir=None):
    """
    Augment a dataset using the VAE+KDE+nearest neighbors approach.
    """
    X = df_train.drop(columns=[target_column])
    y = df_train[target_column]
    print(f'X.shape data augment vae - {X.shape}')
    
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)
    
    n_samples, n_features = X.shape
    
    latent_dim = min(16, max(4, n_features // 4))
    
    if n_samples < 30:
        hidden_dims = [32, 16]
        dropout_rate = 0.2
        beta = 0.05
        balance_factor = 0.5
    elif n_samples < 100:
        hidden_dims = [64, 32]
        dropout_rate = 0.1
        beta = 0.1
        balance_factor = 0.3
    else:
        hidden_dims = [128, 64]
        dropout_rate = 0.1
        beta = 0.1
        balance_factor = 0.2
    
    augmenter = VAEKDEAugmenter(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        num_regions=num_bins,
        balance_strategy=balance_strategy,
        min_samples_per_region=augmented_bucket_size,
        beta=beta,
        dropout_rate=dropout_rate,
        epochs=200,
        early_stopping=20,
        n_neighbors=min(5, n_samples-1),
        use_latent_neighbors=True,
        balance_factor=balance_factor,
        batch_size=min(32, n_samples)
    )
    
    X_augmented, y_augmented = augmenter.augment_dataset(X, y, bin_edges, region_assignments)
    
    df_augmented = X_augmented.copy()
    df_augmented[target_column] = y_augmented
    
    if save_dir is not None:
        augmenter.vae.eval()
        with torch.no_grad():
            original_tensor = torch.FloatTensor(augmenter.X_scaled).to(augmenter.device)
            mu, _ = augmenter.vae.encode(original_tensor)
            latent_space = mu.cpu().numpy()
        
        latent_cols = [f"latent_dim_{i}" for i in range(augmenter.latent_dim)]
        latent_df = pd.DataFrame(latent_space, columns=latent_cols)
        latent_df[target_column] = y.values
        latent_df.to_csv(os.path.join(save_dir, "latent_space.csv"), index=False)
        print(f"Saved latent space representation to {os.path.join(save_dir, 'latent_space.csv')}")
    
    return df_augmented


def create_augmenter_for_mmli(n_samples):
    """
    Create a VAE-KDE augmenter configured for the MMLI dataset (small datasets ~43 samples)
    
    Parameters:
    -----------
    n_samples : int
        Original dataset size
        
    Returns:
    --------
    augmenter : SimpleVAEKDEAugmenter
        Configured augmenter instance that generates exact number of synthetic samples
    """
    # Configure for very small dataset
    if n_samples < 30:
        hidden_dims = [32, 16]
        dropout_rate = 0.2
        beta = 0.05
        balance_factor = 0.5
        latent_dim = 8
    elif n_samples < 100:
        hidden_dims = [64, 32]
        dropout_rate = 0.1
        beta = 0.1
        balance_factor = 0.3
        latent_dim = 12
    else:
        hidden_dims = [128, 64]
        dropout_rate = 0.1
        beta = 0.1
        balance_factor = 0.2
        latent_dim = 16
    
    augmenter = SimpleVAEKDEAugmenter(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta=beta,
        dropout_rate=dropout_rate,
        epochs=200,
        early_stopping=20,
        n_neighbors=min(5, n_samples - 1),
        balance_factor=balance_factor,
        batch_size=min(32, n_samples)
    )
    
    return augmenter


class SimpleVAEKDEAugmenter:
    """
    Simplified VAE-KDE Augmenter that generates an EXACT number of synthetic samples.
    This is the version used by run_experiment.py for the MMLI sample size experiment.
    """
    def __init__(
        self,
        latent_dim=16,
        hidden_dims=[64, 32],
        beta=0.1,
        noise_factor=0.05,
        learning_rate=0.001,
        batch_size=32,
        epochs=200,
        early_stopping=20,
        n_neighbors=5,
        dropout_rate=0.1,
        balance_factor=0.3,
        device=None,
        random_state=13,
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.noise_factor = noise_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.n_neighbors = n_neighbors
        self.dropout_rate = dropout_rate
        self.balance_factor = balance_factor
        
        if device is None:
            # Check for GPU availability: CUDA (NVIDIA) or MPS (Apple M1/M2)
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.vae = None
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def fit(self, X, y):
        """Fit the VAE on the training data"""
        print(f"Using device: {self.device}")
        
        if isinstance(X, pd.DataFrame):
            self.X_columns = X.columns.tolist()
            X_values = X.values
        else:
            self.X_columns = [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X
            
        if isinstance(y, pd.Series):
            self.y_name = y.name
            y_values = y.values
        else:
            self.y_name = "target"
            y_values = y
        
        self.X_original = X_values.copy()
        self.y_original = y_values.copy()
        self.X_scaled = self.scaler.fit_transform(X_values)
        self.y = y_values
        
        input_dim = self.X_scaled.shape[1]
        self.vae = MolecularVAE(
            input_dim=input_dim, 
            latent_dim=self.latent_dim, 
            hidden_dim1=self.hidden_dims[0],
            hidden_dim2=self.hidden_dims[1],
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        self._train_vae()
        self.z_mean, self.z_log_var = self._encode_data()
        
        n_samples = len(self.X_scaled)
        n_neighbors = min(self.n_neighbors, n_samples - 1)
        print(f"Using {n_neighbors} nearest neighbors for {n_samples} samples")
        
        self.nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(self.z_mean)
        self.distances, self.indices = self.nn.kneighbors()
        
        bandwidth = np.std(self.y) * (len(self.y) ** -0.2)
        self.kde = KernelDensity(bandwidth=bandwidth).fit(self.y.reshape(-1, 1))
        self.density = np.exp(self.kde.score_samples(self.y.reshape(-1, 1)))
        
        return self
    
    def _train_vae(self):
        """Train the VAE model"""
        X_tensor = torch.FloatTensor(self.X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        
        batch_size = min(self.batch_size, len(X_tensor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.vae.train()
            total_loss = 0
            
            for batch in dataloader:
                x = batch[0]
                optimizer.zero_grad()
                
                reconstructed, mu, log_var = self.vae(x)
                recon_loss = F.mse_loss(reconstructed, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + self.beta * kl_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def _encode_data(self):
        """Encode data to latent space"""
        self.vae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X_scaled).to(self.device)
            mu, log_var = self.vae.encode(X_tensor)
            return mu.cpu().numpy(), log_var.cpu().numpy()
    
    def generate_samples(self, n_samples):
        """
        Generate exactly n_samples synthetic samples
        
        Parameters:
        -----------
        n_samples : int
            Exact number of synthetic samples to generate
            
        Returns:
        --------
        X_synthetic : numpy array
            Synthetic feature data (in original scale)
        y_synthetic : numpy array
            Synthetic target values
        """
        print(f"Generating {n_samples} synthetic samples...")
        start_time = time.time()
        
        X_synthetic_list = []
        y_synthetic_list = []
        
        self.vae.eval()
        
        inverse_density = 1.0 / (self.density + 1e-10)
        inverse_density_probs = inverse_density / inverse_density.sum()
        uniform_probs = np.ones(len(self.y)) / len(self.y)
        
        sampling_probs = (1 - self.balance_factor) * inverse_density_probs + self.balance_factor * uniform_probs
        sampling_probs = sampling_probs / sampling_probs.sum()
        
        for i in range(n_samples):
            idx = np.random.choice(len(self.y), p=sampling_probs)
            neighbor_idx = np.random.choice(self.indices[idx, 1:])
            
            mu1 = self.z_mean[idx]
            mu2 = self.z_mean[neighbor_idx]
            
            lambd = np.random.uniform(0.2, 0.8)
            z_new = lambd * mu1 + (1 - lambd) * mu2
            z_new += np.random.normal(0, self.noise_factor, size=z_new.shape)
            
            with torch.no_grad():
                z_new_tensor = torch.FloatTensor(z_new.reshape(1, -1)).to(self.device)
                X_synthetic = self.vae.decode(z_new_tensor).cpu().numpy()[0]
            
            X_synthetic_list.append(X_synthetic)
            y_synthetic = lambd * self.y[idx] + (1 - lambd) * self.y[neighbor_idx]
            y_synthetic_list.append(y_synthetic)
            
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{n_samples} samples...")
        
        X_synthetic_array = np.array(X_synthetic_list)
        y_synthetic_array = np.array(y_synthetic_list)
        X_synthetic_original = self.scaler.inverse_transform(X_synthetic_array)
        
        end_time = time.time()
        print(f"Generated {n_samples} synthetic samples in {end_time - start_time:.2f} seconds")
        
        return X_synthetic_original, y_synthetic_array
    
    def augment_dataset(self, X, y, n_synthetic_samples):
        """
        Fit the model and generate synthetic samples in one step
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features
        y : pandas Series or numpy array
            The target variable
        n_synthetic_samples : int
            Number of synthetic samples to generate
                
        Returns:
        --------
        X_augmented : pandas DataFrame
            Original and synthetic samples combined
        y_augmented : pandas Series or numpy array
            Original and synthetic targets combined
        is_synthetic : numpy array
            Boolean array indicating which samples are synthetic
        """
        self.fit(X, y)
        X_synthetic, y_synthetic = self.generate_samples(n_synthetic_samples)
        
        if isinstance(X, pd.DataFrame):
            X_synth_df = pd.DataFrame(X_synthetic, columns=X.columns)
            X_augmented = pd.concat([X.reset_index(drop=True), X_synth_df], ignore_index=True)
        else:
            X_augmented = np.vstack([X, X_synthetic])
        
        if isinstance(y, pd.Series):
            y_augmented = pd.concat([y.reset_index(drop=True), pd.Series(y_synthetic, name=y.name)], ignore_index=True)
        else:
            y_augmented = np.concatenate([y, y_synthetic])
        
        is_synthetic = np.concatenate([
            np.zeros(len(y), dtype=bool),
            np.ones(len(y_synthetic), dtype=bool)
        ])
        
        return X_augmented, y_augmented, is_synthetic
