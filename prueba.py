def _custom_centroids(self) -> np.ndarray:
    #minims i maxims de cada dimensio (D,)
    min_vals = self.X.min(axis=0)
    max_vals = self.X.max(axis=0)

    #k punts equispaiats entre 0 i 1
    t = np.linspace(0, 1, self.K)   #[0, 1/(K-1), 2/(K-1), ..., 1]

    #recorrem diagonal
    centroids = min_vals + t[:, None] * (max_vals - min_vals)  #(K, D)

    return np.array(centroids, dtype=np.float64)