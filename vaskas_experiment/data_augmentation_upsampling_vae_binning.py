import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time
import warnings
warnings.filterwarnings("ignore")

class MolecularVAE(nn.Module):
    """
    Variational Autoencoder for molecular descriptors with 2-layer architecture
    """
    def __init__(self, input_dim, latent_dim=16, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.1):
        super(MolecularVAE, self).__init__()
        
        # Two-layer encoder with dropout for regularization
        self.encoder_layer1 = nn.Linear(input_dim, hidden_dim1)
        self.encoder_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Mean and variance layers
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim2, latent_dim)
        
        # Two-layer decoder
        self.decoder_layer1 = nn.Linear(latent_dim, hidden_dim2)
        self.decoder_layer2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.decoder_output = nn.Linear(hidden_dim1, input_dim)
        
    def encode(self, x):
        """
        Encode input to mu and log_var in latent space
        """
        h1 = F.relu(self.encoder_layer1(x))
        h1 = self.dropout(h1)
        h2 = F.relu(self.encoder_layer2(h1))
        
        # Get latent distribution parameters
        mu = self.fc_mu(h2)
        log_var = self.fc_var(h2)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + std * eps
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector back to input space
        """
        h1 = F.relu(self.decoder_layer1(z))
        h2 = F.relu(self.decoder_layer2(h1))
        reconstructed = self.decoder_output(h2)
        return reconstructed
    
    def forward(self, x):
        """
        Forward pass through the VAE
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var

class DataAugmentationVAE:
    """
    implementation using VAE and K-means for molecular data augmentation
    """
    def __init__(
        self,
        latent_dim=16,            # Dimension of latent space
        hidden_dims=[64, 32],     # Hidden layer dimensions for encoder/decoder
        num_bins=10,              # Number of bins for target variable balancing
        balance_strategy='equal', # Strategy for balancing: 'equal', 'proportional', 'increase'
        augmented_bucket_size='None', # Minimum target count for each bin when using 'increase' strategy
        beta=0.1,                 # Weight of KL divergence term in VAE loss
        noise_factor=0.05,        # Additional noise to add to latent vectors
        learning_rate=0.001,      # Learning rate for VAE training
        batch_size=32,            # Batch size for VAE training
        epochs=200,               # Maximum training epochs
        early_stopping=20,        # Early stopping patience
        min_clusters=2,           # Minimum number of clusters
        dropout_rate=0.1,         # Dropout rate for VAE
        cluster_latent=True,      # Whether to cluster in latent space
        n_init=10,                # Number of initializations for K-means
        device=None,              # PyTorch device (CPU/GPU)
        random_state=42           # Random seed for reproducibility
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_bins = num_bins
        self.balance_strategy = balance_strategy
        self.augmented_bucket_size = augmented_bucket_size
        self.beta = beta
        self.noise_factor = noise_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.min_clusters = min_clusters
        self.dropout_rate = dropout_rate
        self.cluster_latent = cluster_latent
        self.n_init = n_init
        
        # Set device (GPU if available, otherwise CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.vae = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def fit(self, X, y, bin_edges=None, y_bins=None):
        """
        Fit the VAE and K-means models
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features (molecular descriptors)
        y : pandas Series or numpy array
            The target variable
            
        Returns:
        --------
        self : object
            Returns self
        """
        print(f"Using device: {self.device}")        
        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_values)
        
        # Build the VAE
        input_dim = X_scaled.shape[1]
        self.vae = MolecularVAE(
            input_dim=input_dim, 
            latent_dim=self.latent_dim, 
            hidden_dim1=self.hidden_dims[0],
            hidden_dim2=self.hidden_dims[1],
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Train the VAE
        self._train_vae(X_scaled)
        
        # Get latent representations
        z_mean, z_log_var = self._encode_data(X_scaled)
        
        # Determine appropriate cluster count based on data size
        n_samples = len(X_scaled)
        if n_samples < 20:
            n_clusters = max(self.min_clusters, n_samples // 5)
        elif n_samples < 50:
            n_clusters = max(self.min_clusters, n_samples // 8)
        else:
            n_clusters = max(self.min_clusters, n_samples // 10)
            
        print(f"Using {n_clusters} clusters for {n_samples} samples")
        
        # Apply K-means clustering
        if self.cluster_latent:
            # Cluster in latent space
            print("Clustering in VAE latent space")
            cluster_data = z_mean
        else:
            # Cluster in original feature space (as in original paper)
            print("Clustering in original feature space")
            cluster_data = X_scaled
            
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=self.n_init
        )
        
        self.cluster_labels = self.kmeans.fit_predict(cluster_data)

        # Use provided bin information or create new bins
        if bin_edges is not None and y_bins is not None:
            # Use provided bins
            self.bin_edges = bin_edges
            self.y_bins = y_bins
            print("Using pre-existing bin information from stratified split")
        else:
            # Create new bins (original code)
            self.bin_edges = np.percentile(y_values, np.linspace(0, 100, self.num_bins + 1))
            
            # Ensure unique bin edges
            if len(np.unique(self.bin_edges)) < len(self.bin_edges):
                self.bin_edges = np.linspace(np.min(y_values), np.max(y_values), self.num_bins + 1)
            
            # Assign samples to bins
            self.y_bins = np.digitize(y_values, self.bin_edges[1:-1])
            print("Created new bin information for augmentation")
        
        # Get bin counts for upsampling planning
        self.bin_counts = np.bincount(self.y_bins, minlength=self.num_bins)
        
        # Plan balanced augmentation
        if self.balance_strategy == 'equal':
            # Make all bins have the same count as the largest bin
            self.target_counts = np.full_like(self.bin_counts, np.max(self.bin_counts))
        elif self.balance_strategy == 'proportional':
            # Make minority bins at least 50% of majority bin
            max_count = np.max(self.bin_counts)
            self.target_counts = np.maximum(self.bin_counts, int(max_count * 0.5))
        elif self.balance_strategy == 'increase':
            # Make all bins have at least the specified target minimum count
            # But never less than the maximum naturally occurring bin count
            if self.augmented_bucket_size is None:
                raise ValueError("increase must be specified when using 'increase' strategy")
            
            max_natural_count = np.max(self.bin_counts)
            effective_target = max(self.augmented_bucket_size, max_natural_count)
            self.target_counts = np.full_like(self.bin_counts, effective_target)
        else:
            # Default to equal
            self.target_counts = np.full_like(self.bin_counts, np.max(self.bin_counts))
        
        # Calculate how many samples to generate for each bin
        self.samples_to_generate = np.maximum(0, self.target_counts - self.bin_counts)
        
        print("Bin distribution before augmentation:")
        for i in range(self.num_bins):
            bin_range = f"{self.bin_edges[i]:.2f} to {self.bin_edges[i+1]:.2f}"
            print(f"  Bin {i+1} ({bin_range}): {self.bin_counts[i]} samples → Target: {self.target_counts[i]}")
        
        # Store the original data and labels for later use
        self.X_scaled = X_scaled
        self.X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.y = y_values
        
        return self
    
    def _train_vae(self, X_scaled):
        """
        Train the VAE model
        """
        print(f"Training VAE with {self.latent_dim} latent dimensions")
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)  # Input = Target for VAE
        dataloader = DataLoader(
            dataset, 
            batch_size=min(self.batch_size, len(X_scaled)), 
            shuffle=True
        )
        
        # Define optimizer
        optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.vae.train()
            epoch_loss = 0.0
            
            for batch_X, _ in dataloader:
                # Forward pass
                reconstructed, mu, log_var = self.vae(batch_X)
                
                # Calculate loss
                # Reconstruction loss
                recon_loss = F.mse_loss(reconstructed, batch_X, reduction='sum')
                
                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Total loss with beta weighting for KL term
                loss = recon_loss + self.beta * kl_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Calculate average loss for epoch
            epoch_loss /= len(X_scaled)
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                # Save best model
                best_model_state = self.vae.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.vae.load_state_dict(best_model_state)
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.6f}")
    
    def _encode_data(self, X_scaled):
        """
        Encode data to latent space
        """
        self.vae.eval()
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            mu, log_var = self.vae.encode(X_tensor)
            mu = mu.cpu().numpy()
            log_var = log_var.cpu().numpy()
            
        return mu, log_var
    
    def generate_samples(self):
        """
        Generate synthetic samples through latent space interpolation
        with balanced upsampling
        
        Returns:
        --------
        X_synthetic : numpy array
            Synthetic feature data
        y_synthetic : numpy array
            Synthetic target values
        """
        print("Generating synthetic samples with balanced upsampling...")
        start_time = time.time()
        
        # Initialize lists for synthetic data
        X_synthetic_list = []
        y_synthetic_list = []
        
        # Set VAE to evaluation mode
        self.vae.eval()
        
        # Track generated samples per bin for monitoring
        generated_per_bin = np.zeros(self.num_bins, dtype=int)
        
        # For each target bin that needs upsampling
        for bin_idx in range(self.num_bins):
            if self.samples_to_generate[bin_idx] <= 0:
                print(f"Bin {bin_idx+1}: No upsampling needed")
                continue
                
            # Get indices of samples in this bin
            bin_indices = np.where(self.y_bins == bin_idx)[0]
            
            if len(bin_indices) < 2:
                print(f"Bin {bin_idx+1}: Not enough samples for interpolation (need at least 2)")
                continue
                
            n_synthetic = self.samples_to_generate[bin_idx]
            print(f"Bin {bin_idx+1}: Generating {n_synthetic} synthetic samples")
            
            # For each cluster, try to generate samples for this bin
            bin_samples_remaining = n_synthetic
            
            for cluster_id in range(self.kmeans.n_clusters):
                # Get indices of samples that are both in this cluster and in this bin
                cluster_bin_indices = [idx for idx in bin_indices 
                                     if self.cluster_labels[idx] == cluster_id]
                
                # Skip if cluster doesn't have enough samples in this bin
                if len(cluster_bin_indices) < 2:
                    continue
                    
                # Calculate how many samples to generate from this cluster for this bin
                # Distribute proportionally to the number of samples in the cluster
                cluster_proportion = len(cluster_bin_indices) / len(bin_indices)
                cluster_samples = min(
                    int(np.ceil(n_synthetic * cluster_proportion)),
                    bin_samples_remaining
                )
                
                # Generate synthetic samples
                for _ in range(cluster_samples):
                    # Randomly select two samples from the same cluster and bin
                    idx1, idx2 = np.random.choice(cluster_bin_indices, 2, replace=False)
                    
                    # Get their latent representations
                    with torch.no_grad():
                        mu1, log_var1 = self.vae.encode(self.X_tensor[idx1].unsqueeze(0))
                        mu2, log_var2 = self.vae.encode(self.X_tensor[idx2].unsqueeze(0))
                        
                        # Convert to numpy for interpolation
                        mu1 = mu1.cpu().numpy()[0]
                        mu2 = mu2.cpu().numpy()[0]
                    
                    # Generate interpolation parameter
                    lambd = np.random.uniform(0, 1)
                    
                    # Interpolate in latent space
                    z_new = lambd * mu1 + (1 - lambd) * mu2
                    
                    # Add small random noise to avoid exact duplicates
                    z_new += np.random.normal(0, self.noise_factor, size=z_new.shape)
                    
                    # Decode to get synthetic sample
                    with torch.no_grad():
                        z_new_tensor = torch.FloatTensor(z_new.reshape(1, -1)).to(self.device)
                        X_synthetic = self.vae.decode(z_new_tensor).cpu().numpy()[0]
                    
                    X_synthetic_list.append(X_synthetic)
                    
                    # Interpolate target value
                    y_synthetic = lambd * self.y[idx1] + (1 - lambd) * self.y[idx2]
                    y_synthetic_list.append(y_synthetic)
                    
                    # Update counters
                    bin_samples_remaining -= 1
                    generated_per_bin[bin_idx] += 1
                    
                    if bin_samples_remaining <= 0:
                        break
                
                if bin_samples_remaining <= 0:
                    break
            
            # If we couldn't generate enough samples using clusters, use random pairs
            if bin_samples_remaining > 0:
                print(f"  Bin {bin_idx+1}: Generating {bin_samples_remaining} additional samples with random pairs")
                
                for _ in range(bin_samples_remaining):
                    # Randomly select any two samples from this bin
                    idx1, idx2 = np.random.choice(bin_indices, 2, replace=False)
                    
                    # Get their latent representations
                    with torch.no_grad():
                        mu1, log_var1 = self.vae.encode(self.X_tensor[idx1].unsqueeze(0))
                        mu2, log_var2 = self.vae.encode(self.X_tensor[idx2].unsqueeze(0))
                        
                        # Convert to numpy for interpolation
                        mu1 = mu1.cpu().numpy()[0]
                        mu2 = mu2.cpu().numpy()[0]
                    
                    # Generate interpolation parameter
                    lambd = np.random.uniform(0, 1)
                    
                    # Interpolate in latent space
                    z_new = lambd * mu1 + (1 - lambd) * mu2
                    
                    # Add small random noise to avoid exact duplicates
                    z_new += np.random.normal(0, self.noise_factor, size=z_new.shape)
                    
                    # Decode to get synthetic sample
                    with torch.no_grad():
                        z_new_tensor = torch.FloatTensor(z_new.reshape(1, -1)).to(self.device)
                        X_synthetic = self.vae.decode(z_new_tensor).cpu().numpy()[0]
                    
                    X_synthetic_list.append(X_synthetic)
                    
                    # Interpolate target value
                    y_synthetic = lambd * self.y[idx1] + (1 - lambd) * self.y[idx2]
                    y_synthetic_list.append(y_synthetic)
                    
                    # Update counter
                    generated_per_bin[bin_idx] += 1
        
        # If no samples were generated (all bins already balanced), generate some random samples
        if len(X_synthetic_list) == 0:
            print("All bins already balanced. Generating some random samples for diversity.")
            n_samples = len(self.X_scaled) // 2  # Generate 50% more samples
            
            for _ in range(n_samples):
                # Randomly select any two samples
                idx1, idx2 = np.random.choice(len(self.X_scaled), 2, replace=False)
                
                # Get their latent representations
                with torch.no_grad():
                    mu1, log_var1 = self.vae.encode(self.X_tensor[idx1].unsqueeze(0))
                    mu2, log_var2 = self.vae.encode(self.X_tensor[idx2].unsqueeze(0))
                    
                    # Convert to numpy for interpolation
                    mu1 = mu1.cpu().numpy()[0]
                    mu2 = mu2.cpu().numpy()[0]
                
                # Generate interpolation parameter
                lambd = np.random.uniform(0, 1)
                
                # Interpolate in latent space
                z_new = lambd * mu1 + (1 - lambd) * mu2
                
                # Add small random noise
                z_new += np.random.normal(0, self.noise_factor, size=z_new.shape)
                
                # Decode to get synthetic sample
                with torch.no_grad():
                    z_new_tensor = torch.FloatTensor(z_new.reshape(1, -1)).to(self.device)
                    X_synthetic = self.vae.decode(z_new_tensor).cpu().numpy()[0]
                
                X_synthetic_list.append(X_synthetic)
                
                # Interpolate target value
                y_synthetic = lambd * self.y[idx1] + (1 - lambd) * self.y[idx2]
                y_synthetic_list.append(y_synthetic)
        
        # Convert lists to arrays
        X_synthetic_array = np.array(X_synthetic_list)
        y_synthetic_array = np.array(y_synthetic_list)
        
        # Inverse transform to get back to original feature scale
        X_synthetic_original = self.scaler.inverse_transform(X_synthetic_array)
        
        # Calculate new bin distribution
        y_combined = np.concatenate([self.y, y_synthetic_array])
        y_bins_combined = np.digitize(y_combined, self.bin_edges[1:-1])
        bin_counts_combined = np.bincount(y_bins_combined, minlength=self.num_bins)
        
        print("\nBin distribution after augmentation:")
        for i in range(self.num_bins):
            bin_range = f"{self.bin_edges[i]:.2f} to {self.bin_edges[i+1]:.2f}"
            original = self.bin_counts[i]
            added = generated_per_bin[i]
            new_total = bin_counts_combined[i]
            print(f"  Bin {i+1} ({bin_range}): {original} + {added} = {new_total} samples")
        
        end_time = time.time()
        print(f"Generated {len(X_synthetic_list)} synthetic samples in {end_time - start_time:.2f} seconds")
        
        return X_synthetic_original, y_synthetic_array
    
    def augment_dataset(self, X, y, bin_edges=None, bin_assignments=None):
        """
        Fit the model and generate synthetic samples in one step
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features
        y : pandas Series or numpy array
            The target variable
        bin_edges : numpy array, optional
            Pre-existing bin edges from stratified split
        y_bins : numpy array, optional
            Pre-existing bin assignments from stratified split
                
        Returns:
        --------
        X_augmented : same type as X
            Original and synthetic samples combined
        y_augmented : same type as y
            Original and synthetic targets combined
        """
        # Fit the model
        self.fit(X, y, bin_edges, bin_assignments)
        
        # Generate synthetic samples
        X_synthetic, y_synthetic = self.generate_samples()
        print(f'x synthetic = {X_synthetic}')
        #X["is_synthetic"] = 0
        X["target_bucket"] = bin_assignments
        # Create synthetic dataframe
        #X_synth_df = pd.DataFrame(X_synthetic, columns=X.columns.drop(['is_synthetic', 'target_bucket']))
        X_synth_df = pd.DataFrame(X_synthetic, columns=X.columns.drop(['target_bucket']))
        #X_synth_df['is_synthetic'] = 1
        X_synth_df['target_bucket'] = np.digitize(y_synthetic, self.bin_edges[1:-1])
        
        # Combine
        X_augmented = pd.concat([X, X_synth_df], ignore_index=True)
            
        # Combine original and synthetic targets
        if isinstance(y, pd.Series):
            y_augmented = pd.concat([y, pd.Series(y_synthetic, name=y.name)], ignore_index=True)
        else:
            y_augmented = np.concatenate([y, y_synthetic])
            
        return X_augmented, y_augmented


# Simple utility function for integration with your regression platform
def data_augment_vae(df_train, target_column, bin_edges, num_bins=10, balance_strategy='equal', augmented_bucket_size=None, bin_assignments=None, save_dir=None):
    """
    Augment a dataset using the VAE+K-means approach with pre-existing bins
    
    Parameters:
    -----------
    df_train : pandas DataFrame
        Input training dataframe with features and target
    target_column : str
        Name of the target column
    bin_edges : numpy array
        Bin edges from stratified split
    num_bins : int, default=10
        Number of bins (should match the number of bins used in stratified split)
    balance_strategy : str, default='equal'
        Strategy for balancing: 'equal', 'proportional', or 'increase'
    augmented_bucket_size : int, optional
        Minimum target count for each bin when using 'target_minimum' strategy.
        Actual target will be max(target_minimum, largest_natural_bin_count)
        
        
    Returns:
    --------
    df_augmented : pandas DataFrame
        Augmented dataframe with original and synthetic samples
    """
    # Separate features and target
    X = df_train.drop(columns=[target_column])
    y = df_train[target_column]
    print(f'X.shape data augment vae - {X.shape}')
    # Get bin assignments if available
    #y_bins = df_train['target_bucket'].values if 'target_bucket' in df_train.columns else None
    
    # Determine appropriate parameters based on data size
    n_samples, n_features = X.shape
    
    # Set latent dimension based on dataset characteristics
    latent_dim = min(16, max(4, n_features // 4))
    
    # Adjust VAE complexity based on dataset size
    if n_samples < 30:
        hidden_dims = [32, 16]
        dropout_rate = 0.2
        beta = 0.05  # Lower KL weight for very small datasets
    elif n_samples < 100:
        hidden_dims = [64, 32]
        dropout_rate = 0.1
        beta = 0.1
    else:
        hidden_dims = [128, 64]
        dropout_rate = 0.1
        beta = 0.1
    
    # Create data augmenter
    augmenter = DataAugmentationVAE(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        num_bins=num_bins,
        balance_strategy=balance_strategy,
        augmented_bucket_size=augmented_bucket_size,
        beta=beta,
        dropout_rate=dropout_rate,
        epochs=200,
        early_stopping=20,
        min_clusters=2,
        batch_size=min(32, n_samples)
    )
    
    # Augment the dataset
    X_augmented, y_augmented = augmenter.augment_dataset(X, y, bin_edges, bin_assignments)

    
    # Recombine features and target
    df_augmented = X_augmented.copy()
    df_augmented[target_column] = y_augmented

    if save_dir is not None:
        # Get the latent representation of the original data
        augmenter.vae.eval()
        with torch.no_grad():
            original_tensor = torch.FloatTensor(augmenter.X_scaled).to(augmenter.device)
            mu, _ = augmenter.vae.encode(original_tensor)
            latent_space = mu.cpu().numpy()
        
        # Create DataFrame with latent dimensions
        latent_cols = [f"latent_dim_{i}" for i in range(augmenter.latent_dim)]
        latent_df = pd.DataFrame(latent_space, columns=latent_cols)
        
        # Add target and region information
        latent_df[target_column] = y.values
        
        # Save to CSV in the iteration directory
        latent_df.to_csv(os.path.join(save_dir, "latent_space.csv"), index=False)
        
        # Print confirmation
        print(f"Saved latent space representation to {os.path.join(save_dir, 'latent_space.csv')}")
   
    
    return df_augmented