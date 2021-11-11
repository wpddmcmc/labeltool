import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

def make_edges(shape):
    n_x=shape[0]
    n_y=shape[1]
    vertices = np.arange(n_x * n_y).reshape((n_x, n_y))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_right, edges_down))
    return edges

def compute_gradients(data):
    l_x, l_y= data.shape
    gr_right = np.abs(data[:, :-1] - data[:, 1:]).ravel()
    gr_down = np.abs(data[:-1] - data[1:]).ravel()
    return np.r_[gr_right, gr_down]

def computer_weight(data, beta=130, eps=1.e-6):
    # beta Penalization coefficient for the random walker motion
    l_x, l_y = data.shape
    gradients = compute_gradients(data)**2
    beta /= 10 * data.std()
    gradients *= beta
    weights = np.exp(- gradients)
    weights += eps
    return weights

def make_laplacian_sparse(edges, weights):
    pixel_nb = edges.max() + 1
    diag = np.arange(pixel_nb)
    i_indices = np.hstack((edges[0], edges[1]))
    j_indices = np.hstack((edges[1], edges[0]))
    data = np.hstack((-weights, -weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)), 
                            shape=(pixel_nb, pixel_nb))
    connect = - np.ravel(lap.sum(axis=1))
    lap = sparse.coo_matrix((np.hstack((data, connect)),
                (np.hstack((i_indices,diag)), np.hstack((j_indices, diag)))), 
                shape=(pixel_nb, pixel_nb))
    return lap.tocsr()
    
def build_laplacian(data):
    edges = make_edges(data.shape)
    weights = computer_weight(data,beta=130, eps=1.e-6)
    lap = make_laplacian_sparse(edges,weights)
    del edges, weights
    return lap

def buildAB(lap_sparse, labels):
    labels = labels[labels>=0]
    indices = np.arange(labels.size) 
    unlabeled_indices = indices[labels == 0] 
    seeds_indices = indices[labels > 0] 
    print("seeds:",seeds_indices)
    # The following two lines take most of the time
    B = lap_sparse[unlabeled_indices][:, seeds_indices]
    lap_sparse = lap_sparse[unlabeled_indices][:, unlabeled_indices]
    nlabels = labels.max()
    rhs = []
    for lab in range(1, nlabels+1):
        mask = labels[seeds_indices] == lab
        #print("lab-",lab,"mask",mask)
        fs = sparse.csr_matrix(mask)
        fs = fs.transpose()
        #print("lab-",lab,"fs",fs)
        rhs.append(B * fs)
    return lap_sparse,rhs

def solve_bf(lap_sparse, B): 
    lap_sparse = lap_sparse.tocsc()
    solver = factorized(lap_sparse.astype(np.double))
    X = np.array([solver(np.array((-B[i]).todense()).ravel())\
            for i in range(len(B))])
    X = np.argmax(X, axis=0)
    \
    return X
    
def _clean_labels_ar(X, labels):
    labels = np.ravel(labels)
    labels[labels == 0] = X
    return labels