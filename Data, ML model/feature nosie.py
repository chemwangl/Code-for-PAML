import pandas as pd
import numpy as np
n_samples = n_example
n_features = 211
X_nose = pd.DataFrame(
    rng.rand(n_samples, n_features),         
    columns=[f'random_{i}' for i in range(n_features)],
    index=df.index                            
)