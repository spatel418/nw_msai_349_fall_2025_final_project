####STARTING DIAGNOSIS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity

def diagnose_kde_augmentation(vae_augmenter, save_path=None):
    """
    Diagnose issues with KDE-based augmentation in the DataAugmentationVAE class
    
    Parameters:
    -----------
    vae_augmenter : DataAugmentationVAE
        An instance of the DataAugmentationVAE class that has been fit to data
    save_path : str, optional
        Path to save diagnostic plots
    
    Returns:
    --------
    dict
        Diagnostic information
    """
    # Check if the augmenter has been fit
    if vae_augmenter.vae is None or not hasattr(vae_augmenter, 'region_edges'):
        raise ValueError("The VAE augmenter must be fit to data first.")
    
    # Extract key information
    y = vae_augmenter.y
    region_edges = vae_augmenter.region_edges
    region_counts = vae_augmenter.region_counts
    density = vae_augmenter.density
    num_regions = vae_augmenter.num_regions
    
    # Create figure for diagnostics
    plt.figure(figsize=(15, 12))
    
    # 1. Original data distribution
    plt.subplot(3, 1, 1)
    counts, bins, _ = plt.hist(y, bins=region_edges, alpha=0.6, color='blue')
    
    # Add region boundaries
    for edge in region_edges:
        plt.axvline(x=edge, color='red', linestyle='--', alpha=0.5)
    
    # Add region labels
    for i in range(num_regions):
        region_min, region_max = region_edges[i], region_edges[i+1]
        region_center = (region_min + region_max) / 2
        plt.text(region_center, max(counts)*0.8, f"Region {i+1}\n{region_counts[i]} samples", 
                 horizontalalignment='center', fontsize=8, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title('Original Data Distribution')
    plt.xlabel('Target Value')
    plt.ylabel('Count')
    
    # 2. KDE and Inverse Density
    plt.subplot(3, 1, 2)
    
    # Plot original data points
    plt.plot(y, np.zeros_like(y), '|', color='blue', ms=10, alpha=0.6)
    
    # Create a grid for KDE visualization
    x_grid = np.linspace(min(region_edges) - 0.1*(max(region_edges)-min(region_edges)), 
                         max(region_edges) + 0.1*(max(region_edges)-min(region_edges)), 
                         1000)
    
    # Re-fit KDE on the grid
    y_reshaped = y.reshape(-1, 1)
    kde = vae_augmenter.kde
    log_dens = kde.score_samples(x_grid.reshape(-1, 1))
    grid_density = np.exp(log_dens)
    
    # Scale for visualization
    grid_density = grid_density / np.max(grid_density) * max(counts)
    
    # Plot KDE
    plt.plot(x_grid, grid_density, 'g-', label='KDE Density')
    
    # Calculate inverse density for sampling priority
    inverse_density = 1.0 / (grid_density + 1e-10)
    inverse_density = inverse_density / np.max(inverse_density) * max(counts)
    
    # Plot inverse density
    plt.plot(x_grid, inverse_density, 'r-', label='Inverse Density (Sampling Priority)')
    
    # Add region boundaries
    for edge in region_edges:
        plt.axvline(x=edge, color='red', linestyle='--', alpha=0.5)
    
    plt.title('KDE Density vs Inverse Density (Sampling Priority)')
    plt.xlabel('Target Value')
    plt.ylabel('Density / Priority')
    plt.legend()
    
    # 3. Synthetic Sample Generation Analysis
    plt.subplot(3, 1, 3)
    
    # Run a simulation of the sample generation process
    n_simulations = 1000
    y_synthetic_sim = []
    source_indices = []
    
    for region_idx in range(num_regions):
        if vae_augmenter.samples_to_generate[region_idx] <= 0:
            continue
            
        # Get region boundaries
        region_min = region_edges[region_idx]
        region_max = region_edges[region_idx + 1]
        region_center = (region_min + region_max) / 2
        region_width = region_max - region_min
        
        # Calculate distance of each sample to this region's center
        distances_to_center = np.abs(y - region_center) / region_width
        
        # Create region-focused probabilities
        region_probs = np.exp(-distances_to_center * 3)
        
        # Create inverse density probabilities
        inverse_density_points = 1.0 / (density + 1e-10)
        inverse_density_probs = inverse_density_points / inverse_density_points.sum()
        
        # Create uniform sampling probabilities
        uniform_probs = np.ones_like(density) / len(density)
        
        # Blend for final sampling probabilities
        base_probs = (1 - vae_augmenter.balance_factor) * inverse_density_probs + vae_augmenter.balance_factor * uniform_probs
        
        # Multiply by region focus
        combined_probs = region_probs * base_probs
        sampling_probs = combined_probs / combined_probs.sum()
        
        # Simulate sample selection (just for target values)
        n_samples = min(vae_augmenter.samples_to_generate[region_idx], n_simulations // num_regions)
        
        for _ in range(n_samples):
            # Select a seed sample
            idx = np.random.choice(range(len(y)), p=sampling_probs)
            
            # Select one of its nearest neighbors
            neighbor_idx = np.random.choice(vae_augmenter.indices[idx, 1:])
            
            # Generate interpolation parameter
            lambd = np.random.uniform(0.2, 0.8)
            
            # Interpolate target value
            y_synthetic = lambd * y[idx] + (1 - lambd) * y[neighbor_idx]
            y_synthetic_sim.append(y_synthetic)
            source_indices.append((idx, neighbor_idx))
    
    # Convert to array
    if y_synthetic_sim:
        y_synthetic_sim = np.array(y_synthetic_sim)
        
        # Plot synthetic samples
        plt.hist(y_synthetic_sim, bins=region_edges, alpha=0.6, color='purple', label='Synthetic Samples')
        
        # Add original distribution outline
        plt.hist(y, bins=region_edges, alpha=0.2, color='blue', label='Original Samples')
        
        # Add region boundaries
        for edge in region_edges:
            plt.axvline(x=edge, color='red', linestyle='--', alpha=0.5)
        
        # Count synthetic samples per region
        synthetic_regions = np.digitize(y_synthetic_sim, region_edges[1:-1])
        synthetic_counts = np.bincount(synthetic_regions, minlength=num_regions)
        
        # Add region labels
        for i in range(num_regions):
            region_min, region_max = region_edges[i], region_edges[i+1]
            region_center = (region_min + region_max) / 2
            plt.text(region_center, max(counts)*0.5, f"Generated\n{synthetic_counts[i]} samples", 
                    horizontalalignment='center', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7))
    else:
        plt.text(0.5, 0.5, "No synthetic samples generated in simulation", 
                horizontalalignment='center', fontsize=12)
    
    plt.title('Simulated Synthetic Sample Distribution')
    plt.xlabel('Target Value')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
    
    # Print summary
    print("KDE Parameter Testing Summary:")
    print("-" * 50)
    print(f"{'Bandwidth':<15} {'Log-Likelihood':<20} {'Region Score'}")
    print("-" * 50)
    
    for bw in bandwidths:
        # Fit KDE
        kde = KernelDensity(bandwidth=bw).fit(y_reshaped)
        log_likelihood = np.mean(kde.score_samples(y_reshaped))
        
        # Calculate density on grid
        log_dens = kde.score_samples(x_grid.reshape(-1, 1))
        grid_density = np.exp(log_dens)
        
        # Scale for visualization
        grid_density = grid_density / np.max(grid_density) * max(counts)
        
        # Calculate inverse density
        inverse_density = 1.0 / (grid_density + 1e-10)
        inverse_density = inverse_density / np.max(inverse_density) * max(counts)
        
        # Calculate region scores
        region_scores = []
        
        for j in range(num_regions):
            region_min, region_max = region_edges[j], region_edges[j+1]
            
            # Count samples in this region
            in_region = (y >= region_min) & (y <= region_max)
            n_in_region = np.sum(in_region)
            
            # Calculate density statistics for this region
            region_grid = (x_grid >= region_min) & (x_grid <= region_max)
            
            if np.any(region_grid):
                region_density = grid_density[region_grid]
                region_inv_density = inverse_density[region_grid]
                
                if n_in_region < np.mean(np.bincount(np.digitize(y, region_edges[1:-1]), minlength=num_regions)):
                    # This is an underrepresented region
                    score = np.mean(region_inv_density) / max(counts)
                else:
                    # This is a well-represented region
                    score = np.mean(region_density) / max(counts)
                
                region_scores.append(score)
        
        overall_score = np.mean(region_scores) if region_scores else 0
        print(f"{bw:<15.4f} {log_likelihood:<20.4f} {overall_score:.4f}")
    
    # Return the results
    return {
        'bandwidths': bandwidths,
        'x_grid': x_grid,
        'region_edges': region_edges
    }

def simulate_augmentation_process(y, region_edges, bandwidth=None, balance_factor=0.3, n_samples_per_region=50, save_path=None):
    """
    Simulate the entire augmentation process with different parameters
    
    Parameters:
    -----------
    y : numpy array
        Target values
    region_edges : numpy array
        Region boundaries
    bandwidth : float, optional
        KDE bandwidth (if None, use Scott's rule)
    balance_factor : float, default=0.3
        Balance between inverse density and uniform sampling
    n_samples_per_region : int, default=50
        Number of samples to generate per region
    save_path : str, optional
        Path to save diagnostic plot
    """
    # Set up bandwidth
    if bandwidth is None:
        bandwidth = 1.06 * np.std(y) * (len(y) ** -0.2)
    
    # Prepare KDE
    y_reshaped = y.reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth).fit(y_reshaped)
    density = np.exp(kde.score_samples(y_reshaped))
    
    # Calculate number of regions
    num_regions = len(region_edges) - 1
    
    # Assign samples to regions
    region_assignments = np.digitize(y, region_edges[1:-1])
    region_counts = np.bincount(region_assignments, minlength=num_regions)
    
    # Set up nearest neighbors
    from sklearn.neighbors import NearestNeighbors
    n_neighbors = min(5, len(y)-1)
    nn = NearestNeighbors(n_neighbors=n_neighbors+1).fit(y_reshaped)
    distances, indices = nn.kneighbors()
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot original data distribution
    plt.subplot(2, 1, 1)
    counts, bins, _ = plt.hist(y, bins=region_edges, alpha=0.6, color='blue', label='Original Samples')
    
    # Add region boundaries
    for edge in region_edges:
        plt.axvline(x=edge, color='red', linestyle='--', alpha=0.5)
    
    # Add region labels
    for i in range(num_regions):
        region_min, region_max = region_edges[i], region_edges[i+1]
        region_center = (region_min + region_max) / 2
        plt.text(region_center, max(counts)*0.8, f"Region {i+1}\n{region_counts[i]} samples", 
                 horizontalalignment='center', fontsize=8, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(f'Original Data Distribution (bandwidth={bandwidth:.4f}, balance_factor={balance_factor})')
    plt.ylabel('Count')
    plt.legend()
    
    # Simulate synthetic sample generation
    plt.subplot(2, 1, 2)
    
    # Initialize lists for synthetic data
    y_synthetic_list = []
    
    # Track generated samples per region
    generated_per_region = np.zeros(num_regions, dtype=int)
    
    # For each target region
    for region_idx in range(num_regions):
        # Get region boundaries
        region_min = region_edges[region_idx]
        region_max = region_edges[region_idx + 1]
        region_center = (region_min + region_max) / 2
        region_width = region_max - region_min
        
        # Number of samples to generate for this region
        n_synthetic = n_samples_per_region
        
        # Calculate distance of each sample to this region's center
        distances_to_center = np.abs(y - region_center) / region_width
        
        # Create region-focused probabilities (exponential decay with distance)
        region_probs = np.exp(-distances_to_center * 3)
        
        # Create inverse density probabilities (for targeting underrepresented areas)
        inverse_density = 1.0 / (density + 1e-10)
        inverse_density_probs = inverse_density / inverse_density.sum()
        
        # Create uniform sampling probabilities for balance
        uniform_probs = np.ones_like(density) / len(density)
        
        # Blend for final sampling probabilities
        base_probs = (1 - balance_factor) * inverse_density_probs + balance_factor * uniform_probs
        
        # Multiply by region focus
        combined_probs = region_probs * base_probs
        sampling_probs = combined_probs / combined_probs.sum()
        
        # Generate samples for this region
        for _ in range(n_synthetic):
            # Select a seed sample based on probabilities
            idx = np.random.choice(range(len(y)), p=sampling_probs)
            
            # Select one of its nearest neighbors
            neighbor_idx = np.random.choice(indices[idx, 1:])
            
            # Generate interpolation parameter
            lambd = np.random.uniform(0.2, 0.8)
            
            # Interpolate target value
            y_synthetic = lambd * y[idx] + (1 - lambd) * y[neighbor_idx]
            y_synthetic_list.append(y_synthetic)
            
            # Update counter
            synthetic_region = np.digitize([y_synthetic], region_edges[1:-1])[0]
            generated_per_region[synthetic_region] += 1
    
    # Convert to array
    y_synthetic_array = np.array(y_synthetic_list)
    
    # Plot synthetic distribution
    plt.hist(y_synthetic_array, bins=region_edges, alpha=0.6, color='green', label='Synthetic Samples')
    
    # Add region boundaries
    for edge in region_edges:
        plt.axvline(x=edge, color='red', linestyle='--', alpha=0.5)
    
    # Add region labels
    for i in range(num_regions):
        region_min, region_max = region_edges[i], region_edges[i+1]
        region_center = (region_min + region_max) / 2
        plt.text(region_center, max(counts)*0.8, f"Generated\n{generated_per_region[i]} samples", 
                 horizontalalignment='center', fontsize=8, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot combined distribution (original + synthetic)
    y_combined = np.concatenate([y, y_synthetic_array])
    combined_regions = np.digitize(y_combined, region_edges[1:-1])
    combined_counts = np.bincount(combined_regions, minlength=num_regions)
    
    plt.hist(y_combined, bins=region_edges, alpha=0.3, color='purple', label='Combined Distribution')
    
    plt.title('Simulated Synthetic and Combined Distribution')
    plt.xlabel('Target Value')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
    
    # Print summary
    print("Augmentation Simulation Summary:")
    print("-" * 60)
    print(f"{'Region':<10} {'Range':<25} {'Original':<10} {'Synthetic':<10} {'Combined':<10}")
    print("-" * 60)
    
    for i in range(num_regions):
        region_min, region_max = region_edges[i], region_edges[i+1]
        region_range = f"{region_min:.2f} to {region_max:.2f}"
        
        print(f"{i+1:<10} {region_range:<25} {region_counts[i]:<10} {generated_per_region[i]:<10} {combined_counts[i]:<10}")
    
    # Calculate uniformity metrics
    original_uniformity = 1.0 - np.std(region_counts) / np.mean(region_counts)
    synthetic_uniformity = 1.0 - np.std(generated_per_region) / np.mean(generated_per_region)
    combined_uniformity = 1.0 - np.std(combined_counts) / np.mean(combined_counts)
    
    print("\nUniformity Metrics (higher is more uniform):")
    print(f"Original Distribution: {original_uniformity:.4f}")
    print(f"Synthetic Distribution: {synthetic_uniformity:.4f}")
    print(f"Combined Distribution: {combined_uniformity:.4f}")
    # return ({
    #     'y_synthetic': y_synthetic_array,
    #     'generated_per_region': generated_per_region,
    #     'combined_counts': combined_counts,
    #     'uniformity': {
    #         'original': original_uniformity,
    #         'synthetic': synthetic_uniformity,
    #         'combined': combined_uniformity
    #     }
    # }, save_path, dpi=300)

    
    plt.show()
    
    # 4. Additional diagnostic information
    print("\nDiagnostic Information:")
    print("-" * 50)
    
    # Check bandwidth
    bandwidth = vae_augmenter.kde.bandwidth
    print(f"KDE bandwidth: {bandwidth:.4f}")
    
    # Calculate Scott's rule reference bandwidth
    scott_bandwidth = 1.06 * np.std(y) * (len(y) ** -0.2)
    print(f"Scott's rule bandwidth: {scott_bandwidth:.4f}")
    
    # Check if KDE is capturing underrepresented regions effectively
    print("\nRegion-specific KDE effectiveness:")
    for i in range(num_regions):
        region_min, region_max = region_edges[i], region_edges[i+1]
        
        # Get samples in this region
        in_region = (y >= region_min) & (y <= region_max)
        n_in_region = np.sum(in_region)
        
        # Get average density in this region
        avg_density = np.mean(density[in_region]) if n_in_region > 0 else 0
        
        # Get inverse density (sampling priority)
        avg_inv_density = 1.0 / (avg_density + 1e-10) if avg_density > 0 else 0
        
        print(f"Region {i+1} ({region_min:.2f} to {region_max:.2f}): {n_in_region} samples")
        print(f"  Avg density: {avg_density:.4f}, Avg sampling priority: {avg_inv_density:.4f}")
    
    # Check interpolation effectiveness
    print("\nInterpolation analysis:")
    
    # Count how many sample pairs can interpolate into each region
    interpolation_counts = np.zeros(num_regions)
    
    for i in range(len(y)):
        for j in vae_augmenter.indices[i, 1:]:  # Skip self
            # Check if interpolation between these points could hit each region
            y1, y2 = y[i], y[j]
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            for region_idx in range(num_regions):
                region_min, region_max = region_edges[region_idx], region_edges[region_idx+1]
                
                # Check if interpolation range overlaps with region
                if (min_y <= region_max and max_y >= region_min):
                    interpolation_counts[region_idx] += 1
                    break
    
    for i in range(num_regions):
        print(f"Region {i+1}: {interpolation_counts[i]} potential interpolation pairs")
    
    # Analyze sampling probability distribution
    if y_synthetic_sim:
        # Count how many synthetic samples actually fall in their target regions
        on_target = 0
        for region_idx in range(num_regions):
            region_min, region_max = region_edges[region_idx], region_edges[region_idx+1]
            
            # Get synthetic samples that were targeting this region
            region_indices = np.arange(len(y_synthetic_sim))[
                synthetic_regions == region_idx]
            
            # Count how many are actually in this region
            in_region = np.sum((y_synthetic_sim[region_indices] >= region_min) & 
                              (y_synthetic_sim[region_indices] <= region_max))
            
            if len(region_indices) > 0:
                on_target_pct = in_region / len(region_indices) * 100
            else:
                on_target_pct = 0
                
            print(f"Region {i+1}: {in_region}/{len(region_indices)} samples on target ({on_target_pct:.1f}%)")
            on_target += in_region
        
        print(f"\nOverall targeting accuracy: {on_target/len(y_synthetic_sim)*100:.1f}%")
    
    # Return diagnostic data
    return {
        'original_distribution': {
            'y': y,
            'region_edges': region_edges,
            'region_counts': region_counts
        },
        'kde': {
            'bandwidth': bandwidth,
            'scott_bandwidth': scott_bandwidth,
            'density': density
        },
        'synthetic_simulation': {
            'y_synthetic': y_synthetic_sim if y_synthetic_sim else None,
            'source_indices': source_indices if y_synthetic_sim else None
        }
    }


def analyze_neighbor_interpolation(vae_augmenter, region_idx, n_samples=20, save_path=None):
    """
    Analyze how neighbor interpolation works for a specific region
    
    Parameters:
    -----------
    vae_augmenter : DataAugmentationVAE
        An instance of the DataAugmentationVAE class that has been fit to data
    region_idx : int
        Index of the region to analyze
    n_samples : int, default=20
        Number of sample pairs to analyze
    save_path : str, optional
        Path to save diagnostic plot
    """
    # Check if the augmenter has been fit
    if vae_augmenter.vae is None or not hasattr(vae_augmenter, 'region_edges'):
        raise ValueError("The VAE augmenter must be fit to data first.")
    
    # Extract key information
    y = vae_augmenter.y
    region_edges = vae_augmenter.region_edges
    region_min = region_edges[region_idx]
    region_max = region_edges[region_idx + 1]
    region_center = (region_min + region_max) / 2
    region_width = region_max - region_min
    
    # Calculate distance of each sample to this region's center
    distances_to_center = np.abs(y - region_center) / region_width
    
    # Create region-focused probabilities
    region_probs = np.exp(-distances_to_center * 3)
    
    # Create inverse density probabilities
    inverse_density = 1.0 / (vae_augmenter.density + 1e-10)
    inverse_density_probs = inverse_density / inverse_density.sum()
    
    # Create uniform sampling probabilities
    uniform_probs = np.ones_like(vae_augmenter.density) / len(vae_augmenter.density)
    
    # Blend for final sampling probabilities
    base_probs = (1 - vae_augmenter.balance_factor) * inverse_density_probs + vae_augmenter.balance_factor * uniform_probs
    
    # Multiply by region focus
    combined_probs = region_probs * base_probs
    sampling_probs = combined_probs / combined_probs.sum()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot original data distribution
    plt.subplot(2, 1, 1)
    plt.hist(y, bins=region_edges, alpha=0.6, color='blue')
    
    # Highlight target region
    plt.axvspan(region_min, region_max, alpha=0.2, color='red')
    
    # Add region boundaries
    for edge in region_edges:
        plt.axvline(x=edge, color='black', linestyle='--', alpha=0.5)
    
    plt.title(f'Region {region_idx+1} ({region_min:.2f} to {region_max:.2f}) Analysis')
    plt.xlabel('Target Value')
    plt.ylabel('Count')
    
    # Analyze neighbor interpolation
    plt.subplot(2, 1, 2)
    
    # Plot all data points
    plt.plot(y, np.zeros_like(y), '|', color='blue', ms=10, alpha=0.3)
    
    # Select sample points based on sampling probabilities
    seed_indices = np.random.choice(range(len(y)), size=n_samples, p=sampling_probs)
    
    # Plot seed points and their neighbors
    for i, idx in enumerate(seed_indices):
        # Get a random neighbor
        neighbor_idx = np.random.choice(vae_augmenter.indices[idx, 1:])
        
        # Get values
        y1, y2 = y[idx], y[neighbor_idx]
        
        # Plot the pair
        y_pos = 0.1 + (i % 10) * 0.08  # Staggered vertical positions
        plt.plot([y1, y2], [y_pos, y_pos], '-', color='green', alpha=0.5)
        plt.plot(y1, y_pos, 'o', color='red', alpha=0.7)
        plt.plot(y2, y_pos, 'o', color='orange', alpha=0.7)
        
        # Simulate interpolation
        lambdas = np.linspace(0.2, 0.8, 7)
        interp_y = np.array([lambd * y1 + (1 - lambd) * y2 for lambd in lambdas])
        
        # Check which interpolated points fall in the target region
        in_region = (interp_y >= region_min) & (interp_y <= region_max)
        
        # Plot interpolated points
        plt.plot(interp_y[~in_region], np.full_like(interp_y[~in_region], y_pos), 'x', color='gray', alpha=0.5)
        plt.plot(interp_y[in_region], np.full_like(interp_y[in_region], y_pos), '*', color='green', alpha=0.8)
        
        # Add labels
        if i < 5:  # Only label the first few for clarity
            min_y, max_y = min(y1, y2), max(y1, y2)
            span_region = (min_y <= region_min and max_y >= region_min) or \
                          (min_y <= region_max and max_y >= region_max) or \
                          (min_y >= region_min and max_y <= region_max)
                          
            plt.text(max_y + region_width*0.05, y_pos, 
                     f"Pair {i+1}: {span_region and 'Can' or 'Cannot'} reach region", 
                     fontsize=8, verticalalignment='center')
    
    # Highlight target region
    plt.axvspan(region_min, region_max, alpha=0.2, color='red')
    
    # Add region boundaries
    for edge in region_edges:
        plt.axvline(x=edge, color='black', linestyle='--', alpha=0.5)
    
    plt.ylim(-0.1, 1.0)
    plt.title('Neighbor Interpolation Analysis')
    plt.xlabel('Target Value')
    plt.ylabel('Sample Pairs')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
    
    # Count how many sample pairs can interpolate into the target region
    n_can_interpolate = 0
    
    # Check all samples
    for i in range(len(y)):
        for j in vae_augmenter.indices[i, 1:]:  # Skip self
            # Check if interpolation between these points could hit the target region
            y1, y2 = y[i], y[j]
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            # Check if interpolation range overlaps with region
            if (min_y <= region_max and max_y >= region_min):
                n_can_interpolate += 1
                break
    
    print(f"\nRegion {region_idx+1} ({region_min:.2f} to {region_max:.2f}) Interpolation Analysis:")
    print(f"Out of {len(y)} samples, {n_can_interpolate} have neighbors that allow interpolation into this region")
    print(f"Interpolation potential: {n_can_interpolate/len(y)*100:.1f}%")
    
    # Check region-specific density statistics
    in_region = (y >= region_min) & (y <= region_max)
    n_in_region = np.sum(in_region)
    
    print(f"\nRegion has {n_in_region} original samples")
    
    if n_in_region > 0:
        avg_density = np.mean(vae_augmenter.density[in_region])
        avg_inv_density = 1.0 / (avg_density + 1e-10)
        print(f"Average density in region: {avg_density:.4f}")
        print(f"Average sampling priority in region: {avg_inv_density:.4f}")
    else:
        print("No samples in this region to calculate density statistics")

def analyze_balance_factor_impact(vae_augmenter, region_idx, save_path=None):
    """
    Analyze how the balance_factor parameter affects sampling probabilities
    
    Parameters:
    -----------
    vae_augmenter : DataAugmentationVAE
        An instance of the DataAugmentationVAE class that has been fit to data
    region_idx : int
        Index of the region to analyze
    save_path : str, optional
        Path to save diagnostic plot
    """
    # Check if the augmenter has been fit
    if vae_augmenter.vae is None or not hasattr(vae_augmenter, 'region_edges'):
        raise ValueError("The VAE augmenter must be fit to data first.")
    
    # Extract key information
    y = vae_augmenter.y
    region_edges = vae_augmenter.region_edges
    region_min = region_edges[region_idx]
    region_max = region_edges[region_idx + 1]
    region_center = (region_min + region_max) / 2
    region_width = region_max - region_min
    
    # Calculate distance of each sample to this region's center
    distances_to_center = np.abs(y - region_center) / region_width
    
    # Create region-focused probabilities
    region_probs = np.exp(-distances_to_center * 3)
    
    # Create inverse density probabilities
    inverse_density = 1.0 / (vae_augmenter.density + 1e-10)
    inverse_density_probs = inverse_density / inverse_density.sum()
    
    # Create uniform sampling probabilities
    uniform_probs = np.ones_like(vae_augmenter.density) / len(vae_augmenter.density)
    
    # Test different balance factors
    balance_factors = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot original data distribution
    plt.subplot(len(balance_factors) + 1, 1, 1)
    counts, bins, _ = plt.hist(y, bins=region_edges, alpha=0.6, color='blue')
    
    # Highlight target region
    plt.axvspan(region_min, region_max, alpha=0.2, color='red')
    
    # Add region boundaries
    for edge in region_edges:
        plt.axvline(x=edge, color='black', linestyle='--', alpha=0.5)
    
    plt.title(f'Region {region_idx+1} ({region_min:.2f} to {region_max:.2f}) Analysis')
    plt.ylabel('Count')
    
    # Plot sampling probabilities for different balance factors
    for i, bf in enumerate(balance_factors):
        plt.subplot(len(balance_factors) + 1, 1, i + 2)
        
        # Blend for final sampling probabilities
        base_probs = (1 - bf) * inverse_density_probs + bf * uniform_probs
        
        # Multiply by region focus
        combined_probs = region_probs * base_probs
        sampling_probs = combined_probs / combined_probs.sum()
        
        # Scale for visualization
        scaled_probs = sampling_probs / np.max(sampling_probs) * max(counts)
        
        # Plot the sampling probabilities
        plt.bar(y, scaled_probs, width=region_width/20, alpha=0.6, color='green')
        
        # Highlight target region
        plt.axvspan(region_min, region_max, alpha=0.2, color='red')
        
        # Add region boundaries
        for edge in region_edges:
            plt.axvline(x=edge, color='black', linestyle='--', alpha=0.5)
        
        # Calculate statistics
        in_region = (y >= region_min) & (y <= region_max)
        prob_in_region = np.sum(sampling_probs[in_region])
        
        plt.title(f'balance_factor = {bf}: {prob_in_region*100:.1f}% probability in target region')
        plt.ylabel('Sampling Probability')
    
    plt.xlabel('Target Value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def test_custom_kde_parameters(y, region_edges, bandwidths=None, save_path=None):
    """
    Test different KDE parameters to find optimal settings for augmentation
    
    Parameters:
    -----------
    y : numpy array
        Target values
    region_edges : numpy array
        Region boundaries
    bandwidths : list, optional
        List of bandwidths to test (if None, automatically select based on data)
    save_path : str, optional
        Path to save diagnostic plot
    """
    # Create a grid for KDE visualization
    x_grid = np.linspace(min(region_edges) - 0.1*(max(region_edges)-min(region_edges)), 
                        max(region_edges) + 0.1*(max(region_edges)-min(region_edges)), 
                        1000)
    
    # Determine bandwidths to test
    if bandwidths is None:
        # Calculate Scott's rule bandwidth
        scott_bw = 1.06 * np.std(y) * (len(y) ** -0.2)
        
        # Test a range around Scott's rule
        bandwidths = [
            scott_bw * 0.1,  # Very narrow
            scott_bw * 0.3,  # Narrow
            scott_bw,        # Scott's rule
            scott_bw * 3,    # Wide
            scott_bw * 10    # Very wide
        ]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot original data distribution
    plt.subplot(len(bandwidths) + 1, 1, 1)
    counts, bins, _ = plt.hist(y, bins=region_edges, alpha=0.6, color='blue')
    
    # Add region boundaries
    for edge in region_edges:
        plt.axvline(x=edge, color='red', linestyle='--', alpha=0.5)
    
    plt.title('Original Data Distribution')
    plt.ylabel('Count')
    
    # Calculate number of regions
    num_regions = len(region_edges) - 1
    
    # Test each bandwidth
    y_reshaped = y.reshape(-1, 1)
    
    for i, bw in enumerate(bandwidths):
        plt.subplot(len(bandwidths) + 1, 1, i + 2)
        
        # Fit KDE with this bandwidth
        kde = KernelDensity(bandwidth=bw).fit(y_reshaped)
        
        # Calculate density on grid
        log_dens = kde.score_samples(x_grid.reshape(-1, 1))
        grid_density = np.exp(log_dens)
        
        # Scale for visualization
        grid_density = grid_density / np.max(grid_density) * max(counts)
        
        # Plot KDE
        plt.plot(x_grid, grid_density, 'g-', label='KDE Density')
        
        # Calculate inverse density for sampling priority
        inverse_density = 1.0 / (grid_density + 1e-10)
        inverse_density = inverse_density / np.max(inverse_density) * max(counts)
        
        # Plot inverse density
        plt.plot(x_grid, inverse_density, 'r-', label='Inverse Density (Sampling Priority)')
        
        # Add region boundaries
        for edge in region_edges:
            plt.axvline(x=edge, color='red', linestyle='--', alpha=0.5)
        
        # Calculate log-likelihood
        log_likelihood = np.mean(kde.score_samples(y_reshaped))
        
        # Calculate region-specific density effectiveness
        region_scores = []
        
        for j in range(num_regions):
            region_min, region_max = region_edges[j], region_edges[j+1]
            
            # Count samples in this region
            in_region = (y >= region_min) & (y <= region_max)
            n_in_region = np.sum(in_region)
            
            # Calculate density statistics for this region
            region_grid = (x_grid >= region_min) & (x_grid <= region_max)
            
            if np.any(region_grid):
                region_density = grid_density[region_grid]
                region_inv_density = inverse_density[region_grid]
                
                # Quantify how well this bandwidth identifies low-density regions
                # Higher score = better at identifying underrepresented regions
                if n_in_region < np.mean(np.bincount(np.digitize(y, region_edges[1:-1]), minlength=num_regions)):
                    # This is an underrepresented region
                    # Lower density (higher inverse density) is better
                    score = np.mean(region_inv_density) / max(counts)
                else:
                    # This is a well-represented region
                    # Higher density (lower inverse density) is better
                    score = np.mean(region_density) / max(counts)
                
                region_scores.append(score)
        
        # Calculate overall score
        overall_score = np.mean(region_scores) if region_scores else 0
        
        plt.title(f'bandwidth = {bw:.4f}: Log-likelihood = {log_likelihood:.4f}, Region score = {overall_score:.4f}')
        plt.ylabel('Density / Priority')
        plt.legend()
    
    plt.xlabel('Target Value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
    
    # Print summary
    print("KDE Parameter Testing Summary:")
    print("-" * 50)
    print(f"{'Bandwidth':<15} {'Log-Likelihood':<20} {'Region Score'}")
    print("-" * 50)
    
    for bw in bandwidths:
        # Fit KDE
        kde = KernelDensity(bandwidth=bw).fit(y_reshaped)
        log_likelihood = np.mean(kde.score_samples(y_reshaped))
        
        # Calculate density on grid
        log_dens = kde.score_samples(x_grid.reshape(-1, 1))
        grid_density = np.exp(log_dens)
        
        # Scale for visualization
        grid_density = grid_density / np.max(grid_density) * max(counts)
        
        # Calculate inverse density
        inverse_density = 1.0 / (grid_density + 1e-10)
        inverse_density = inverse_density / np.max(inverse_density) * max(counts)
        
        # Calculate region scores
        region_scores = []
        
        for j in range(num_regions):
            region_min, region_max = region_edges[j], region_edges[j+1]
            
            # Count samples in this region
            in_region = (y >= region_min) & (y <= region_max)
            n_in_region = np.sum(in_region)
            
            # Calculate density statistics for this region
            region_grid = (x_grid >= region_min) & (x_grid <= region_max)
            
            if np.any(region_grid):
                region_density = grid_density[region_grid]
                region_inv_density = inverse_density[region_grid]
                
                if n_in_region < np.mean(np.bincount(np.digitize(y, region_edges[1:-1]), minlength=num_regions)):
                    # This is an underrepresented region
                    score = np.mean(region_inv_density) / max(counts)
                else:
                    # This is a well-represented region
                    score = np.mean(region_density) / max(counts)
                
                region_scores.append(score)
        
        overall_score = np.mean(region_scores) if region_scores else 0
        print(f"{bw:<15.4f} {log_likelihood:<20.4f} {overall_score:.4f}")
    
    # Return the results
    return {
        'bandwidths': bandwidths,
        'x_grid': x_grid,
        'region_edges': region_edges
    }