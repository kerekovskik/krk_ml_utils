import jax.numpy as jnp
import jax
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from functools import partial
from typing import List, Tuple, Any, Optional, Generator
import glob

def jax_collate_fn_pad(batch, pad_value=32002, max_len_targets=None, max_len_features=None):
    """
    Collate function that pads sequences to the max length in a batch.
    It handles both features and targets.
    """
    # Separate features and targets
    features = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Find the max length in features and targets for this batch
    max_len_features_batch = max(len(f) for f in features)
    max_len_targets_batch = max(len(t) for t in targets)
    
    if max_len_features is None or max_len_features < max_len_features_batch:
        max_len_features = max_len_features_batch
    
    if max_len_targets is None or max_len_targets < max_len_targets_batch:
        max_len_targets = max_len_targets_batch
        
    
    # Pad each sequence to the max length
    padded_features = np.array([np.pad(f, (0, max_len_features - len(f)), 'constant', constant_values=pad_value) for f in features])
    padded_targets = np.array([np.pad(t, (0, max_len_targets - len(t)), 'constant', constant_values=pad_value) for t in targets])

    # Convert to JAX arrays
    return jnp.array(padded_features), jnp.array(padded_targets)


def jax_collate_fn(batch):
    """Custom collate function that returns JAX arrays instead of PyTorch tensors."""
    features = jnp.array(np.stack([item[0] for item in batch]))
    targets = jnp.array(np.stack([item[1] for item in batch]))
    return features, targets


JaxDataLoader = partial(DataLoader, collate_fn=jax_collate_fn)

def create_jax_nlp_dataloader(pad_value=32002, max_len_targets=None, max_len_features=None):
    """
    Factory function to create a JaxNLPDataLoader with a custom pad value.
    
    Args:
        pad_value (int): Token ID to use for padding sequences. Defaults to 32002.
    
    Returns:
        A partial DataLoader configured with the jax_collate_fn_pad function.
    """
    return partial(DataLoader, collate_fn=partial(
        jax_collate_fn_pad, 
        pad_value=pad_value, 
        max_len_features=max_len_features, max_len_targets=max_len_targets
        ))

# Keep the original for backward compatibility
JaxNLPDataLoader = create_jax_nlp_dataloader()

class NumpyDataset(Dataset):
    def __init__(self, file_path, features_key="x", labels_key="y", rngs=None, allow_pickle=False, preload=False):
        """
        Args:
            file_path (string): Path to the .npz file.
        """
        # np.load is memory-efficient with .npz files as it memory-maps the file
        # by default, not loading the entire arrays into RAM unless accessed.
        data = np.load(file_path, allow_pickle=allow_pickle)

        # Get number of samples
        num_samples = data[features_key].shape[0]

        if rngs is None:
            self.features = data[features_key]
            self.labels = data[labels_key]
        else:
            key = rngs.default()
            # Generate a random permutation of indices
            random_indices = jax.random.randint(
                key, shape=(num_samples,), minval=0, maxval=num_samples - 1
            )
            self.features = data[features_key][random_indices]
            self.labels = data[labels_key][random_indices]
        if preload:
            # Preload the data into memory
            self.features = self.features.copy()
            self.labels = self.labels.copy()
        
    def __len__(self):
        # This returns the total number of samples
        return self.features.shape[0]

    def __getitem__(self, idx):
        # 1. Select the sample at the given index `idx`
        feature_sample = self.features[idx]
        label_sample = self.labels[idx]

        return feature_sample, label_sample


class ShardedNPZDataset(Dataset):
    """A PyTorch Dataset for data sharded across multiple files.

    This class supports two modes of operation based on the `is_npz` flag:

    1.  **NPZ Mode (`is_npz=True`, default):**
        - Expects a directory of .npz files (shards), where each file contains
          both features and targets.
        - Uses numpy's lazy loading to read slices from the compressed or
          uncompressed arrays within each .npz archive.

    2.  **NPY Mode (`is_npz=False`):**
        - Expects a directory of individual .npy files.
        - Requires a specific naming convention: feature files must be named
          like `{features_key}_0.npy`, `{features_key}_1.npy`, etc., and target
          files must be named `{targets_key}_0.npy`, `{targets_key}_1.npy`, etc.
        - Uses memory-mapping (`mmap_mode`) for maximum memory efficiency,
          letting the OS handle on-demand page loading from disk. This is
          ideal for uncompressed data.

    In both modes, the class first scans the directory to build a map of the
    dataset's structure, allowing for efficient global indexing.
    """

    def __init__(
        self,
        directory: str,
        features_key: str = "x",
        targets_key: str = "y",
        is_npz: bool = True,
    ):
        """
        Initializes the dataset by scanning the data directory.

        Args:
            directory (str): The path to the directory containing the data shards.
            features_key (str, optional): The base name for feature files (in
                NPY mode) or the key for the features array (in NPZ mode).
                Defaults to 'x'.
            targets_key (str, optional): The base name for target files (in
                NPY mode) or the key for the targets array (in NPZ mode).
                Defaults to 'y'.
            is_npz (bool, optional): If True, operates in NPZ mode. If False,
                operates in NPY mode with memory-mapping. Defaults to True.
        """
        self.directory = directory
        self.features_key = features_key
        self.targets_key = targets_key
        self.is_npz = is_npz

        self.shard_paths: List[str | Tuple[str, str]] = []
        self.cumulative_samples = []
        self.total_samples = 0

        self._build_index()

    def _build_index(self):
        """Scans the directory and builds an index of samples."""
        print(
            f"Building index for '{self.directory}' in {'NPZ' if self.is_npz else 'NPY'} mode..."
        )

        if self.is_npz:
            # NPZ Mode: Find all .npz files.
            self.shard_paths = sorted(glob.glob(os.path.join(self.directory, "*.npz")))
            if not self.shard_paths:
                raise FileNotFoundError(f"No .npz files found in '{self.directory}'.")

            samples_per_shard = []
            for path in self.shard_paths:
                # Open .npz, read array header to get length without loading data.
                with np.load(path) as data:
                    num_samples = len(data[self.features_key])
                    samples_per_shard.append(num_samples)

        else:
            # NPY Mode: Find and pair feature and target .npy files.
            feature_files = sorted(
                glob.glob(os.path.join(self.directory, f"{self.features_key}_*.npy"))
            )
            target_files = sorted(
                glob.glob(os.path.join(self.directory, f"{self.targets_key}_*.npy"))
            )

            if not feature_files:
                raise FileNotFoundError(
                    f"No feature files matching '{self.features_key}_*.npy' found."
                )
            if len(feature_files) != len(target_files):
                raise ValueError("Mismatch between number of feature and target files.")

            self.shard_paths = list(zip(feature_files, target_files))
            samples_per_shard = []
            for f_path, t_path in self.shard_paths:
                # Memory-map the array to read its shape (very fast).
                features_mmap = np.load(f_path, mmap_mode="r")
                targets_mmap = np.load(t_path, mmap_mode="r")
                if features_mmap.shape[0] != targets_mmap.shape[0]:
                    raise ValueError(
                        f"Sample count mismatch in shard: {f_path} and {t_path}"
                    )
                samples_per_shard.append(features_mmap.shape[0])

        self.cumulative_samples = np.cumsum(samples_per_shard)
        self.total_samples = self.cumulative_samples[-1]
        print(
            f"Index built. Found {self.total_samples} samples across {len(self.shard_paths)} shards."
        )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Fetches the sample at the given global index."""
        if not 0 <= idx < self.total_samples:
            raise IndexError("Index out of range")

        # Find which shard the index belongs to using the cumulative sum.
        shard_idx = np.searchsorted(self.cumulative_samples, idx, side="right")

        # Calculate the local index within that shard.
        local_idx = idx - (
            self.cumulative_samples[shard_idx - 1] if shard_idx > 0 else 0
        )

        if self.is_npz:
            # NPZ Mode: Load the slice from the .npz archive.
            npz_path = self.shard_paths[shard_idx]
            with np.load(npz_path) as data:
                feature_sample = data[self.features_key][local_idx]
                target_sample = data[self.targets_key][local_idx]
        else:
            # NPY Mode: Get the slice from the memory-mapped arrays.
            feature_path, target_path = self.shard_paths[shard_idx]
            features_mmap = np.load(feature_path, mmap_mode="r")
            targets_mmap = np.load(target_path, mmap_mode="r")
            feature_sample = features_mmap[local_idx]
            target_sample = targets_mmap[local_idx]

        # Convert numpy arrays to PyTorch tensors.
        # Use .copy() because arrays from memmap or npz are not writable,
        # which can cause issues with some PyTorch operations/transforms.
        feature_tensor = feature_sample
        target_tensor = target_sample

        return feature_tensor, target_tensor


# Define a clear type hint for our generator
DataGenerator = Generator[Tuple[np.ndarray, np.ndarray], None, None]


def create_sharded_npz_from_generator(
    directory_location: str,
    data_generator: DataGenerator,
    features_key: str = "x",
    targets_key: str = "y",
    compress: bool = True,
):
    """Creates a directory of sharded .npz files from a data generator.

    This function is memory-efficient, processing one chunk of data from the
    generator at a time and saving it directly to a separate .npz file. The
    resulting directory is structured for consumption by a `ShardedNPZDataset`.

    Args:
        directory_location (str): The path to the directory where the sharded
            .npz files will be saved. The directory will be created if it
            does not exist.
        data_generator (Generator): A generator function that yields tuples of
            (features, targets), where both are NumPy arrays. Each yielded
            tuple will be saved as a single shard.
        features_key (str, optional): The key to use for storing the features
            array within each .npz file. Defaults to 'x'.
        targets_key (str, optional): The key to use for storing the targets
            array within each .npz file. Defaults to 'y'.
        compress (bool, optional): If True, uses `np.savez_compressed` to save
            smaller files at the cost of some CPU time. If False, uses
            `np.savez`. Defaults to True.
    """
    # Ensure the target directory exists.
    os.makedirs(directory_location, exist_ok=True)
    print(f"Saving shards to directory: '{directory_location}'")

    # Determine which save function to use based on the compression flag.
    save_func = np.savez_compressed if compress else np.savez

    # Enumerate through the generator to get an index for each shard.
    for i, (x_chunk, y_chunk) in enumerate(data_generator):
        if not isinstance(x_chunk, np.ndarray) or not isinstance(y_chunk, np.ndarray):
            raise TypeError(
                f"Generator yielded non-NumPy array type at iteration {i}. "
                f"Got ({type(x_chunk)}, {type(y_chunk)})."
            )

        # Construct the file path for the current shard.
        shard_path = os.path.join(directory_location, f"shard_{i}.npz")

        # Save the chunk to its own .npz file. The keyword arguments
        # become the keys for the arrays within the file.
        save_func(shard_path, **{features_key: x_chunk, targets_key: y_chunk})
        print(
            f"  - Saved {shard_path} with shapes "
            f"X: {x_chunk.shape}, Y: {y_chunk.shape}"
        )

    print("Finished creating all shards.")


def create_single_npz_from_dataframe(
    file_path: str,
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    label_columns: List[str],
    features_key: str = "x",
    targets_key: str = "y",
    feature_dtype: np.dtype = np.float32,
    target_dtype: Optional[np.dtype] = None,
    compress: bool = True,
):
    """Creates a single .npz file from a pandas DataFrame.

    This function extracts specified feature and label columns from a DataFrame,
    converts them into two NumPy arrays, and saves them into a single .npz
    file suitable for use with the `NPZDataset` class.

    Args:
        file_path (str): The full path for the output .npz file. If it does not
            end with '.npz', the extension will be appended.
        dataframe (pd.DataFrame): The source DataFrame containing the data.
        feature_columns (List[str]): A list of column names to be extracted
            and combined into the features array (X).
        label_columns (List[str]): A list of column names to be extracted
            and combined into the labels array (Y).
        features_key (str, optional): The key to use for storing the features
            array in the .npz file. Defaults to 'x'.
        targets_key (str, optional): The key to use for storing the targets
            array in the .npz file. Defaults to 'y'.
        feature_dtype (np.dtype, optional): The NumPy data type for the
            features array. Defaults to np.float32.
        target_dtype (Optional[np.dtype], optional): The NumPy data type for
            the targets array. If None, it's inferred by NumPy. Defaults to None.
        compress (bool, optional): If True, uses `np.savez_compressed`.
            If False, uses `np.savez`. Defaults to True.

    Raises:
        KeyError: If any of the provided column names do not exist in the
            DataFrame.
    """
    # Ensure the file path has the correct extension.
    if not file_path.endswith(".npz"):
        file_path += ".npz"

    print(f"Creating .npz file at: '{file_path}'")
    try:
        # For sequence data, we expect a single column with list/array objects.
        # We convert the pandas Series to a NumPy array of objects.
        if len(feature_columns) == 1 and isinstance(
            dataframe[feature_columns[0]].iloc[0], (list, np.ndarray)
        ):
            x_array = dataframe[feature_columns[0]].to_numpy()
        else:
            x_array = dataframe[feature_columns].to_numpy(dtype=feature_dtype)

        # Do the same for labels.
        if len(label_columns) == 1 and isinstance(
            dataframe[label_columns[0]].iloc[0], (list, np.ndarray)
        ):
            y_array = dataframe[label_columns[0]].to_numpy()
        else:
            y_array = dataframe[label_columns].to_numpy(dtype=target_dtype)
    except KeyError as e:
        print("Error: A specified column was not found in the DataFrame.")
        raise e

    # Determine which save function to use.
    save_func = np.savez_compressed if compress else np.savez

    # Save the two arrays into the .npz file.
    save_func(file_path, **{features_key: x_array, targets_key: y_array})
    print("  - File saved successfully.")
    print(
        f"  - Features array '{features_key}' shape: {x_array.shape}, dtype: {x_array.dtype}"
    )
    print(
        f"  - Targets array '{targets_key}' shape: {y_array.shape}, dtype: {y_array.dtype}"
    )
