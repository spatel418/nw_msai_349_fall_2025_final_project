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
    Implementation using VAE, KDE, and nearest neighbors for molecular data augmentation
    """
    def __init__(
        self,
        latent_dim=16,            # Dimension of latent space
        hidden_dims=[64, 32],     # Hidden layer dimensions for encoder/decoder
        num_regions=10,           # Number of regions for target variable balancing
        balance_strategy='equal', # Strategy for balancing: 'equal', 'proportional', 'increase'
        min_samples_per_region=None, # Minimum target count for each region (formerly augmented_bucket_size)
        beta=0.1,                 # Weight of KL divergence term in VAE loss
        noise_factor=0.05,        # Additional noise to add to latent vectors
        learning_rate=0.001,      # Learning rate for VAE training
        batch_size=32,            # Batch size for VAE training
        epochs=200,               # Maximum training epochs
        early_stopping=20,        # Early stopping patience
        n_neighbors=5,            # Number of neighbors to consider (formerly min_clusters)
        dropout_rate=0.1,         # Dropout rate for VAE
        use_latent_neighbors=True, # Whether to use latent space for neighbor finding (formerly cluster_latent)
        balance_factor=0.3,       # How much to balance density vs. uniform sampling (new parameter)
        device=None,              # PyTorch device (CPU/GPU)
        random_state=13,           # Random seed for reproducibility
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
    
    def fit(self, X, y, region_edges=None, region_assignments=None):
        """
        Fit the VAE and prepare for KDE-based augmentation
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features (molecular descriptors)
        y : pandas Series or numpy array
            The target variable
        region_edges : numpy array, optional
            Pre-existing region boundaries (formerly bin_edges)
        region_assignments : numpy array, optional
            Pre-existing region assignments (formerly y_bins)
            
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
        
        # Determine appropriate number of neighbors based on data size
        n_samples = len(X_scaled)
        n_neighbors = min(self.n_neighbors, n_samples-1)
            
        print(f"Using {n_neighbors} nearest neighbors for {n_samples} samples")
        
        
        if self.use_latent_neighbors:
            # Find neighbors in latent space
            print("Finding neighbors in VAE latent space")
            neighbor_data = z_mean
        else:
            # Find neighbors in original feature space
            print("Finding neighbors in original feature space")
            neighbor_data = X_scaled
            
        # Use NearestNeighbors instead of KMeans
        self.nn = NearestNeighbors(
            n_neighbors=n_neighbors+1,  # +1 because first neighbor is self
            algorithm='auto'
        ).fit(neighbor_data)
        
        self.distances, self.indices = self.nn.kneighbors()
        
        # Use provided region information or create new regions
        if region_edges is not None and region_assignments is not None:
            # Use provided regions
            self.region_edges = region_edges
            self.region_assignments = region_assignments
            print("Using pre-existing region information from stratified split")
        else:
            ####TODO I can likely delete this
            # Create new regions
            self.region_edges = np.percentile(y_values, np.linspace(0, 100, self.num_regions + 1))
            
            # Ensure unique region edges
            if len(np.unique(self.region_edges)) < len(self.region_edges):
                self.region_edges = np.linspace(np.min(y_values), np.max(y_values), self.num_regions + 1)
            
            # Assign samples to regions
            self.region_assignments = np.digitize(y_values, self.region_edges[1:-1])
            print("Created new region information for augmentation")
        
        # Get region counts for upsampling planning
        self.region_counts = np.bincount(self.region_assignments, minlength=self.num_regions)
        
        # Plan balanced augmentation
        if self.balance_strategy == 'equal':
            # Make all regions have the same count as the largest region
            self.target_counts = np.full_like(self.region_counts, np.max(self.region_counts))
        elif self.balance_strategy == 'proportional':
            # Make minority regions at least 50% of majority region
            max_count = np.max(self.region_counts)
            self.target_counts = np.maximum(self.region_counts, int(max_count * 0.5))
        elif self.balance_strategy == 'increase':
            # Make all regions have at least the specified target minimum count
            # But never less than the maximum naturally occurring region count
            if self.min_samples_per_region is None:
                raise ValueError("min_samples_per_region must be specified when using 'increase' strategy")
            
            max_natural_count = np.max(self.region_counts)
            effective_target = max(self.min_samples_per_region, max_natural_count)
            self.target_counts = np.full_like(self.region_counts, effective_target)
        else:
            # Default to equal
            self.target_counts = np.full_like(self.region_counts, np.max(self.region_counts))
        
        # Calculate how many samples to generate for each region
        self.samples_to_generate = np.maximum(0, self.target_counts - self.region_counts)
        
        print("Region distribution before augmentation:")
        for i in range(self.num_regions):
            region_range = f"{self.region_edges[i]:.2f} to {self.region_edges[i+1]:.2f}"
            print(f"  Region {i+1} ({region_range}): {self.region_counts[i]} samples → Target: {self.target_counts[i]}")
        
        # Store the original data and labels for later use
        self.X_scaled = X_scaled
        self.X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.y = y_values
        
        # Compute KDE for target values
        y_reshaped = y_values.reshape(-1, 1)
        
        # Check KDE quality first
        bandwidth, log_likelihood = self._check_kde_quality(y_values)
        
        self.kde = KernelDensity(bandwidth=float(bandwidth)).fit(y_reshaped)
        self.density = np.exp(self.kde.score_samples(y_reshaped))
        
        return self
    
    def _check_kde_quality(self, y_values, max_iterations=3):
        """
        Bandwidth optimization that tries both increasing and decreasing
        """
        y_reshaped = y_values.reshape(-1, 1)
        
        # Calculate initial bandwidth
        bandwidth = 0.5 * y_values.std() * (len(y_values) ** -0.2)
        if len(y_values) < 30:
            bandwidth *= 1.5
        
        # Fit KDE and check quality
        kde = KernelDensity(bandwidth=bandwidth).fit(y_reshaped)
        log_likelihood = np.mean(kde.score_samples(y_reshaped))
        
        print(f"Initial KDE quality: bandwidth={bandwidth:.4f}, log-likelihood={log_likelihood:.4f}")
        
        # If log-likelihood is already good, no need to adjust
        if log_likelihood >= -3.0:
            return bandwidth, log_likelihood
        
        # Try both increasing and decreasing bandwidth
        bandwidth_increase = bandwidth * 1.5
        kde_increase = KernelDensity(bandwidth=bandwidth_increase).fit(y_reshaped)
        ll_increase = np.mean(kde_increase.score_samples(y_reshaped))
        
        bandwidth_decrease = bandwidth * 0.75
        kde_decrease = KernelDensity(bandwidth=bandwidth_decrease).fit(y_reshaped)
        ll_decrease = np.mean(kde_decrease.score_samples(y_reshaped))
        
        print(f"Increased bandwidth: {bandwidth_increase:.4f}, log-likelihood: {ll_increase:.4f}")
        print(f"Decreased bandwidth: {bandwidth_decrease:.4f}, log-likelihood: {ll_decrease:.4f}")
        
        # Find best option
        options = [
            (bandwidth, log_likelihood),
            (bandwidth_increase, ll_increase),
            (bandwidth_decrease, ll_decrease)
        ]
        
        bandwidth, log_likelihood = max(options, key=lambda x: x[1])
        print(f"Selected bandwidth: {bandwidth:.4f}, log-likelihood: {log_likelihood:.4f}")
        
        # If we haven't reached target quality and have iterations left, continue
        if log_likelihood < -3.0 and max_iterations > 1:
            # Recursive call with remaining iterations (selected bandwidth as starting point)
            return self._check_kde_quality(
                y_values, 
                max_iterations=max_iterations-1
            )
        
        return bandwidth, log_likelihood
    
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
        with KDE-based balanced upsampling
        
        Returns:
        --------
        X_synthetic : numpy array
            Synthetic feature data
        y_synthetic : numpy array
            Synthetic target values
        """
        print("Generating synthetic samples with KDE-based balanced upsampling...")
        start_time = time.time()
        
        # Initialize lists for synthetic data
        X_synthetic_list = []
        y_synthetic_list = []
        
        # Set VAE to evaluation mode
        self.vae.eval()
        
        # Track generated samples per region for monitoring
        generated_per_region = np.zeros(self.num_regions, dtype=int)
        
        # For each target region that needs upsampling
        for region_idx in range(self.num_regions):
            if self.samples_to_generate[region_idx] <= 0:
                print(f"Region {region_idx+1}: No upsampling needed")
                continue
                
            # Get region boundaries
            region_min = self.region_edges[region_idx]
            region_max = self.region_edges[region_idx + 1]
            region_center = (region_min + region_max) / 2
            region_width = region_max - region_min
            
            # Number of samples to generate for this region
            n_synthetic = self.samples_to_generate[region_idx]
            print(f"Region {region_idx+1}: Generating {n_synthetic} synthetic samples")
            
            # Calculate distance of each sample to this region's center
            distances_to_center = np.abs(self.y - region_center) / region_width
            
            # Create region-focused probabilities (exponential decay with distance)
            region_probs = np.exp(-distances_to_center * 3)
            
            # Create inverse density probabilities (for targeting underrepresented areas)
            inverse_density = 1.0 / (self.density + 1e-10)
            inverse_density_probs = inverse_density / inverse_density.sum()
            
            # Create uniform sampling probabilities for balance
            uniform_probs = np.ones_like(self.density) / len(self.density)
            
            # Blend for final sampling probabilities
            # Balance between targeting this region and targeting low-density areas
            base_probs = (1 - self.balance_factor) * inverse_density_probs + self.balance_factor * uniform_probs
            
            # Multiply by region focus
            combined_probs = region_probs * base_probs
            sampling_probs = combined_probs / combined_probs.sum()
            
            # Generate samples for this region
            for _ in range(n_synthetic):
                # Select a seed sample based on our probabilities
                idx = np.random.choice(range(len(self.X_scaled)), p=sampling_probs)
                
                # Select one of its nearest neighbors (excluding self)
                neighbor_idx = np.random.choice(self.indices[idx, 1:])
                
                # Get their latent representations
                with torch.no_grad():
                    mu1, log_var1 = self.vae.encode(self.X_tensor[idx].unsqueeze(0))
                    mu2, log_var2 = self.vae.encode(self.X_tensor[neighbor_idx].unsqueeze(0))
                    
                    # Convert to numpy for interpolation
                    mu1 = mu1.cpu().numpy()[0]
                    mu2 = mu2.cpu().numpy()[0]
                
                # Generate interpolation parameter
                lambd = np.random.uniform(0.2, 0.8)
                
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
                y_synthetic = lambd * self.y[idx] + (1 - lambd) * self.y[neighbor_idx]
                y_synthetic_list.append(y_synthetic)
                
                # Update counter - use digitize to determine which region the synthetic sample belongs to
                synthetic_region = np.digitize([y_synthetic], self.region_edges[1:-1])[0]
                generated_per_region[synthetic_region] += 1
        
        # If no samples were generated (all regions already balanced), generate some random samples
        if len(X_synthetic_list) == 0:
            print("All regions already balanced. Generating some random samples for diversity.")
            n_samples = len(self.X_scaled) // 4  # Generate 25% more samples
            
            for _ in range(n_samples):
                # Randomly select any sample
                idx = np.random.choice(len(self.X_scaled))
                
                # Select one of its nearest neighbors
                neighbor_idx = np.random.choice(self.indices[idx, 1:])
                
                # Get their latent representations
                with torch.no_grad():
                    mu1, log_var1 = self.vae.encode(self.X_tensor[idx].unsqueeze(0))
                    mu2, log_var2 = self.vae.encode(self.X_tensor[neighbor_idx].unsqueeze(0))
                    
                    # Convert to numpy for interpolation
                    mu1 = mu1.cpu().numpy()[0]
                    mu2 = mu2.cpu().numpy()[0]
                
                # Generate interpolation parameter
                lambd = np.random.uniform(0.2, 0.8)
                
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
                y_synthetic = lambd * self.y[idx] + (1 - lambd) * self.y[neighbor_idx]
                y_synthetic_list.append(y_synthetic)
                
                # Update counter
                synthetic_region = np.digitize([y_synthetic], self.region_edges[1:-1])[0]
                generated_per_region[synthetic_region] += 1
        
        # Convert lists to arrays
        X_synthetic_array = np.array(X_synthetic_list)
        y_synthetic_array = np.array(y_synthetic_list)
        
        # Inverse transform to get back to original feature scale
        X_synthetic_original = self.scaler.inverse_transform(X_synthetic_array)
        
        # Calculate new region distribution
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
        """
        Fit the model and generate synthetic samples in one step
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The input features
        y : pandas Series or numpy array
            The target variable
        region_edges : numpy array, optional
            Pre-existing region edges from stratified split (formerly bin_edges)
        region_assignments : numpy array, optional
            Pre-existing region assignments from stratified split (formerly y_bins)
                
        Returns:
        --------
        X_augmented : same type as X
            Original and synthetic samples combined
        y_augmented : same type as y
            Original and synthetic targets combined
        """
        # Fit the model
        self.fit(X, y, region_edges, region_assignments)
        
        # Generate synthetic samples
        X_synthetic, y_synthetic = self.generate_samples()
        print(f'x synthetic = {X_synthetic}')
        
        # Add target_region column to original data (for compatibility)
        X["target_region"] = region_assignments
        
        # Create synthetic dataframe
        X_synth_df = pd.DataFrame(X_synthetic, columns=X.columns.drop(['target_region']))
        
        # Add target_region to synthetic data (for compatibility)
        X_synth_df['target_region'] = np.digitize(y_synthetic, self.region_edges[1:-1])
        
        # Combine
        X_augmented = pd.concat([X, X_synth_df], ignore_index=True)
            
        # Combine original and synthetic targets
        if isinstance(y, pd.Series):
            y_augmented = pd.concat([y, pd.Series(y_synthetic, name=y.name)], ignore_index=True)
        else:
            y_augmented = np.concatenate([y, y_synthetic])
            
        return X_augmented, y_augmented


# Simple utility function for integration with your regression platform
def data_augment_vae(df_train, target_column, bin_edges, num_bins=10, balance_strategy='equal', augmented_bucket_size=None, region_assignments=None, save_dir=None):
    """
    Augment a dataset using the VAE+KDE+nearest neighbors approach
    
    Parameters:
    -----------
    df_train : pandas DataFrame
        Input training dataframe with features and target
    target_column : str
        Name of the target column
    bin_edges : numpy array
        Region boundaries (formerly bin_edges from stratified split)
    num_bins : int, default=10
        Number of regions (should match the number of regions used in stratified split)
    balance_strategy : str, default='equal'
        Strategy for balancing: 'equal', 'proportional', or 'increase'
    augmented_bucket_size : int, optional
        Minimum target count for each region when using 'increase' strategy.
        Renamed to min_samples_per_region in the class implementation.
        
    Returns:
    --------
    df_augmented : pandas DataFrame
        Augmented dataframe with original and synthetic samples
    """
    # Separate features and target
    X = df_train.drop(columns=[target_column])
    y = df_train[target_column]
    print(f'X.shape data augment vae - {X.shape}')
        
    # Determine appropriate parameters based on data size
    n_samples, n_features = X.shape
    
    # Set latent dimension based on dataset characteristics
    latent_dim = min(16, max(4, n_features // 4))
    
    # Adjust VAE complexity based on dataset size
    if n_samples < 30:
        hidden_dims = [32, 16]
        dropout_rate = 0.2
        beta = 0.05  # Lower KL weight for very small datasets
        balance_factor = 0.5  # More uniform sampling for very small datasets
    elif n_samples < 100:
        hidden_dims = [64, 32]
        dropout_rate = 0.1
        beta = 0.1
        balance_factor = 0.3
    else:
        hidden_dims = [128, 64]
        dropout_rate = 0.1
        beta = 0.1
        balance_factor = 0.2  # More density-focused for larger datasets
    
    # Create data augmenter with updated parameter names
    augmenter = DataAugmentationVAE(
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
    
    # Augment the dataset (with updated parameter names)
    X_augmented, y_augmented = augmenter.augment_dataset(X, y, bin_edges, region_assignments)
    
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
