import torch
import numpy as np
import pgeof
import math
from tqdm import tqdm

from src.utils.neighbors import neighbors_dense_to_csr
from src.utils.scatter import scatter_pca


__all__ = [
    'cross_product_matrix',
    'rodrigues_rotation_matrix',
    'base_vectors_3d',
    'geometric_features']


def cross_product_matrix(k):
    """Compute the cross-product matrix of a vector k.

    Credit: https://github.com/torch-points3d/torch-points3d
    """
    return torch.tensor(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=k.device)


def rodrigues_rotation_matrix(axis, theta_degrees):
    """Given an axis and a rotation angle, compute the rotation matrix
    using the Rodrigues formula.

    Source : https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    Credit: https://github.com/torch-points3d/torch-points3d
    """
    axis = axis / axis.norm()
    K = cross_product_matrix(axis)
    t = torch.tensor([theta_degrees / 180. * np.pi], device=axis.device)
    R = torch.eye(3, device=axis.device) \
        + torch.sin(t) * K + (1 - torch.cos(t)) * K.mm(K)
    return R


def base_vectors_3d(x):
    """Compute orthonormal bases for a set of 3D vectors. The 1st base
    vector is the normalized input vector, while the 2nd and 3rd vectors
    are constructed in the corresponding orthogonal plane. Note that
    this problem is underconstrained and, as such, any rotation of the
    output base around the 1st vector is a valid orthonormal base.
    """
    assert x.dim() == 2
    assert x.shape[1] == 3

    # First direction is along x
    a = x

    # If x is 0 vector (norm=0), arbitrarily put a to (1, 0, 0)
    a[torch.where(a.norm(dim=1) == 0)[0]] = torch.tensor(
        [[1, 0, 0]], dtype=x.dtype, device=x.device)

    # Safely normalize a
    a = a / a.norm(dim=1).view(-1, 1)

    # Build a vector orthogonal to a
    b = torch.vstack((a[:, 1] - a[:, 2], a[:, 2] - a[:, 0], a[:, 0] - a[:, 1])).T

    # In the same fashion as when building a, the second base vector
    # may be 0 by construction (i.e. a is of type (v, v, v)). So we need
    # to deal with this edge case by setting
    b[torch.where(b.norm(dim=1) == 0)[0]] = torch.tensor(
        [[2, 1, -1]], dtype=x.dtype, device=x.device)

    # Safely normalize b
    b /= b.norm(dim=1).view(-1, 1)

    # Cross product of a and b to build the 3rd base vector
    c = torch.linalg.cross(a, b)

    return torch.cat((a.unsqueeze(1), b.unsqueeze(1), c.unsqueeze(1)), dim=1)


def geometric_features(
        xyz,
        nn,
        k_min=5,
        k_step=-1,
        k_min_search=25,
        add_self_as_neighbor=True,
        chunk_size=100000,
        verbose=False,
        algorithm='eigh'):
    device = xyz.device
    N = xyz.shape[0]

    # If required, we add each point to its own neighborhood before
    # computation
    if add_self_as_neighbor:
        nn = torch.cat((torch.arange(N, device=device).view(-1, 1), nn), dim=1)

    # Depending on the device of the input, we will use different
    # approaches for computing the geometric features. If on CPU, we
    # will rely on the pgeof library. If on GPU, we will use a
    # torch-cuda implementation
    if xyz.is_cpu:
        features = geometric_features_pgeof(
            xyz,
            nn,
            k_min=k_min,
            k_step=k_step,
            k_min_search=k_min_search)
    else:
        features = geometric_features_torch(
            xyz,
            nn,
            k_min=k_min,
            k_step=k_step,
            k_min_search=k_min_search,
            chunk_size=chunk_size,
            verbose=verbose,
            algorithm=algorithm)

    # Heuristic to increase importance of verticality in partition
    features['verticality'] *= 2

    # We choose to orient normal vectors towards Z+, by convention
    features['normal'][features['normal'][:, 2] < 0] *= -1

    return features


def geometric_features_pgeof(
        xyz,
        nn,
        k_min=5,
        k_step=-1,
        k_min_search=25):
    device = xyz.device

    # Convert neighbor indices from 2D tensor to CSR format
    # Check for missing neighbors (indicated by -1 indices)
    nn_ptr, nn, _ = neighbors_dense_to_csr(nn)

    # Make sure array are contiguous before moving to C++
    xyz = np.ascontiguousarray(xyz.cpu().numpy())
    nn = np.ascontiguousarray(nn.cpu().numpy().astype('uint32'))
    nn_ptr = np.ascontiguousarray(nn_ptr.cpu().numpy().astype('uint32'))

    # C++ geometric features computation on CPU
    if k_step < 0:
        f = pgeof.compute_features(
            xyz,
            nn,
            nn_ptr,
            k_min,
            verbose=False)
    else:
        f = pgeof.compute_features_optimal(
            xyz,
            nn,
            nn_ptr,
            k_min,
            k_step,
            k_min_search,
            verbose=False)
    f = torch.from_numpy(f).to(device)

    return dict(
        linearity=f[:, 0].view(-1, 1),
        planarity=f[:, 1].view(-1, 1),
        scattering=f[:, 2].view(-1, 1),
        verticality=f[:, 3].view(-1, 1),
        curvature=f[:, 10].view(-1, 1),
        length=f[:, 7].view(-1, 1),
        surface=f[:, 8].view(-1, 1),
        volume=f[:, 9].view(-1, 1),
        normal=f[:, 4:7].view(-1, 3))


def geometric_features_torch(
        xyz,
        nn,
        k_min=5,
        k_step=-1,
        k_min_search=25,
        chunk_size=None,
        verbose=False,
        algorithm='eigh'):
    # Recursive call in case chunk is specified. Chunk allows limiting
    # the number of neighborhoods processed at once. This might
    # alleviate memory use
    N = xyz.shape[0]
    if chunk_size is not None and chunk_size > 0:

        # Recursive call on smaller chunks
        chunk_size = int(chunk_size) if chunk_size > 1 \
            else math.ceil(N * chunk_size)
        num_chunks = math.ceil(N / chunk_size)

        output_list = None
        enum = range(num_chunks) if not verbose else tqdm(range(num_chunks))
        for i_chunk in enum:
            # Only keep the neighborhoods for the chunk at hand
            start = i_chunk * chunk_size
            end = (i_chunk + 1) * chunk_size
            nn_chunk = nn[start:end]

            # PCA on the neighborhoods
            output_chunk = _geometric_features_torch(
                xyz,
                nn_chunk,
                k_min=k_min,
                k_step=k_step,
                k_min_search=k_min_search,
                algorithm=algorithm)

            # Initialize the keys at the first iteration
            if output_list is None:
                output_list = {k: [v] for k, v in output_chunk.items()}
                continue

            # Accumulate results
            for k, v in output_chunk.items():
                output_list[k].append(v)

        output = {k: torch.cat(v, dim=0) for k, v in output_list.items()}
    else:
        output = _geometric_features_torch(
            xyz,
            nn,
            k_min=k_min,
            k_step=k_step,
            k_min_search=k_min_search,
            algorithm=algorithm)

    return output


def _geometric_features_torch(
        xyz,
        nn,
        k_min=5,
        k_step=-1,
        k_min_search=25,
        algorithm='eigh'):
    k_max = nn.shape[1]

    if k_step < 0:
        eigenval, eigenvec, sizes = _scatter_pca_from_dense_neighbors(
            xyz,
            nn,
            algorithm=algorithm)
    else:
        eigenval = None
        eigenvec = None
        sizes = None
        eigenentropy = 1
        k0 = max(k_min, k_min_search)
        for k in range(k0, k_max + 1):

            # Only evaluate the neighborhood's PCA every 'k_step' and at
            # the boundary values
            if (k > k0) and (k % k_step != 0) and (k != k_max):
                continue

            # PCA on the first k neighbors
            # NB: this assumes nn to be sorted by increasing distance
            # along dim=1
            eigenval_k, eigenvec_k, sizes_k = _scatter_pca_from_dense_neighbors(
                xyz,
                nn[:, :k],
                algorithm=algorithm)

            # Compute the eigenentropy of each neighborhood
            epsilon = 1e-3
            e = eigenval_k / (eigenval_k.sum(dim=1).view(-1, 1) + epsilon)
            eigenentropy_k = (-e * torch.log(e + epsilon)).sum(dim=1)

            # Initialization at the first iteration
            if k == k0:
                eigenval = eigenval_k
                eigenvec = eigenvec_k
                sizes = sizes_k
                eigenentropy = eigenentropy_k
                continue

            # Keep track of the eigendecomposition with the lowest
            # eigenentropy
            idx_k = torch.where(eigenentropy_k < eigenentropy)[0]
            eigenval[idx_k] = eigenval_k[idx_k]
            eigenvec[idx_k] = eigenvec_k[idx_k]
            sizes[idx_k] = sizes_k[idx_k]
            eigenentropy[idx_k] = eigenentropy_k[idx_k]

    # The eigenvalues are sorted in INCREASING order. So the normal is
    # the FIRST eigenvector, associated with the smallest eigenvalue
    normal = eigenvec[:, :, 0]

    # Compute the eigenfeatures
    # We align with SPG here:
    # https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/ply_c/ply_c.cpp
    # NB: the eigenvalues are returned sorted in INCREASING order, but
    # we usually tend to consider them in decreasing order for
    # eigenfeatures computation. Don't get lost there!
    lambda_1 = eigenval[:, 2].sqrt()
    lambda_2 = eigenval[:, 1].sqrt()
    lambda_3 = eigenval[:, 0].sqrt()

    linearity = (lambda_1 - lambda_2) / (lambda_1 + 1e-3)
    planarity = (lambda_2 - lambda_3) / (lambda_1 + 1e-3)
    scattering = lambda_3 / (lambda_1 + 1e-3)
    length = lambda_1
    surface = (lambda_1 * lambda_2 + 1e-6).sqrt()
    volume = (lambda_1 * lambda_2 * lambda_3 + 1e-9).pow(1 / 3)
    curvature = lambda_3 / (lambda_1 + lambda_2 + lambda_3 + 1e-3)

    unary = (eigenvec.abs() * eigenval.unsqueeze(1)).sum(dim=2)
    verticality = unary[:, 2] / (unary.norm(dim=1) + 1e-8)

    # Set all features to 0 for all clouds with fewer than k_min
    idx_small = torch.where(sizes < k_min)[0]
    linearity[idx_small] *= 0
    planarity[idx_small] *= 0
    scattering[idx_small] *= 0
    verticality[idx_small] *= 0
    curvature[idx_small] *= 0
    length[idx_small] *= 0
    surface[idx_small] *= 0
    volume[idx_small] *= 0
    normal[idx_small] *= 0

    return dict(
        linearity=linearity.view(-1, 1),
        planarity=planarity.view(-1, 1),
        scattering=scattering.view(-1, 1),
        verticality=verticality.view(-1, 1),
        curvature=curvature.view(-1, 1),
        length=length.view(-1, 1),
        surface=surface.view(-1, 1),
        volume=volume.view(-1, 1),
        normal=normal.view(-1, 3))


def _scatter_pca_from_dense_neighbors(xyz, nn, algorithm='eigh'):
    device = xyz.device
    N = nn.shape[0]

    # Convert neighbor indices from 2D tensor to CSR format
    # Check for missing neighbors (indicated by -1 indices)
    nn_ptr, nn, sizes = neighbors_dense_to_csr(nn)

    # Compute PCA for each neighborhood
    idx = torch.repeat_interleave(
        torch.arange(N, device=device),
        nn_ptr[1:] - nn_ptr[:-1])
    eigenval, eigenvec = scatter_pca(
        xyz[nn],
        idx,
        algorithm=algorithm)

    return eigenval, eigenvec, sizes
