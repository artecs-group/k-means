import time
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

n_centers = 16
iterations = 300

X, _ = make_blobs(n_samples=100000, centers=n_centers, n_features=2)

start = time.time()

k_means = KMeans(n_clusters=n_centers, max_iter=iterations, tol=1e-16, init='random')
k_means.fit(X)
labels = k_means.predict(X)

total_elapsed = time.time() - start
print('Took {0:.8f}s ({1} runs)'.format(total_elapsed, iterations))