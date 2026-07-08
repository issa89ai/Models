import numpy as np
import visualization
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture


def generate_data(n_data, means, covariances, weights):
    """creates a list of data points"""
    n_clusters, n_features = means.shape
    
    data = np.zeros((n_data, n_features))
    for i in range(n_data):
        # pick a cluster id and create data from this cluster
        k = np.random.choice(n_clusters, size = 1, p = weights)[0]
        x = np.random.multivariate_normal(means[k], covariances[k])
        data[i] = x
   
    return data

def testGMMsklearnBICAIC():
     # Model parameters, including the mean
    # covariance matrix and the weights for each cluster
    init_means = np.array([
        [5, 0],
        [1, 1],
        [0, 5]
    ])
    
    init_covariances = np.array([
        [[.5, 0.], [0, .5]],
        [[.92, .38], [.38, .91]],
        [[.5, 0.], [0, .5]]
    ])
    
    init_weights = [1 / 4, 1 / 2, 1 / 4]
    
    # generate data
    np.random.seed(4)
    X = generate_data(100, init_means, init_covariances, init_weights)
    
    #plt.plot(X[:, 0], X[:, 1], 'ko')
    #plt.tight_layout()
    
    n_components = np.arange(1, 10)
    clfs = [GaussianMixture(n, max_iter = 1000).fit(X) for n in n_components]
    bics = [clf.bic(X) for clf in clfs]
    aics = [clf.aic(X) for clf in clfs]
    
    plt.plot(n_components, bics, label = 'BIC')
    plt.plot(n_components, aics, label = 'AIC')
    plt.xlabel('n_components')
    plt.legend()
    plt.show()
## Generate synthetic data
N,D = 1000, 3 # number of points and dimenstinality

if D == 2:
    #set gaussian ceters and covariances in 2D
    means = np.array([[0.5, 0.0],
                      [0, 0],
                      [-0.5, -0.5],
                      [-0.8, 0.3]])
    covs = np.array([np.diag([0.01, 0.01]),
                     np.diag([0.025, 0.01]),
                     np.diag([0.01, 0.025]),
                     np.diag([0.01, 0.01])])
elif D == 3:
    # set gaussian ceters and covariances in 3D
    means = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [-0.5, -0.5, -0.5],
                      [-0.8, 0.3, 0.4]])
    covs = np.array([np.diag([0.01, 0.01, 0.03]),
                     np.diag([0.08, 0.01, 0.01]),
                     np.diag([0.01, 0.05, 0.01]),
                     np.diag([0.03, 0.07, 0.01])])
n_gaussians = means.shape[0]

points = []
for i in range(len(means)):
    x = np.random.multivariate_normal(means[i], covs[i], N )
    points.append(x)
points = np.concatenate(points)

#fit the gaussian model
gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
gmm.fit(points)

#visualize
if D == 2:
    visualization.visualize_2D_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
elif D == 3:
    visualization.visualize_3d_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
    
testGMMsklearnBICAIC()
