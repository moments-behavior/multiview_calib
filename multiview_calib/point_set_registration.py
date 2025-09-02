#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2020
# --------------------------------------------------------------------------
import numpy as np
import itertools
import logging
import cv2

logger = logging.getLogger(__name__)

def average_distance(X, Y):
    return np.linalg.norm(X-Y, axis=1).mean()

def apply_rigid_transform(X, R, t, scale):
    return np.dot(X*scale, R.T) + t[None]

def estimate_scale_point_sets(src, dst, max_est=50000):
    
    idxs = np.arange(len(src))
    np.random.shuffle(idxs)
    
    # computes cross ratios between all pairs of points
    scales = []
    for i, (j,k) in enumerate(itertools.combinations(idxs, 2)):
        d1 = np.linalg.norm(src[j]-src[k])
        d2 = np.linalg.norm(dst[j]-dst[k])
        scales.append(d2/d1)
        
        if i>max_est:
            break
        
    return np.nanmedian(scales), np.nanstd(scales)


def orthogonal_procrustes_modified(A, B, do_reflection=True):
    # this is based on the MATLAB implementation that allows to handle reflections
    
    from scipy.linalg import svd
    
    A = np.asanyarray(A)
    B = np.asanyarray(B)
    
    if A.ndim != 2:
        raise ValueError(f'expected ndim to be 2, but observed {A.ndim}')
    if A.shape != B.shape:
        raise ValueError(f'the shapes of A and B differ ({A.shape} vs {B.shape})')
   
    H = B.T @ A
    u, w, vt = svd(H)
    R = u @ vt
    
    
    if(np.linalg.det(R) < 0 and do_reflection):
        
        logging.info("Reflection needed in orthogonal_procrustes_modified")
        vt[-1, :] *= -1
        w[-1] *= -1
        R = u @ vt
        
    # The scale is the sum of the singular values.
    scale = w.sum()
    
    # The function was missing this return statement.
    return R, scale
    
    scale = w.sum()

def procrustes_registration(src, dst):
    """
    Estimates rotation translation and scale of two point sets
    using Procrustes analysis
    
    dst = (src*scale x R.T) + t + residuals
    
    Parameters:
    ----------
    src : numpy.ndarray (N,3)
        transformed points set
    dst : numpy.ndarray (N,3)
        target points set   
        
    Return:
    -------
    scale, rotation matrix, translation and average distance
    between the alligned points sets
    """
    from scipy.linalg import orthogonal_procrustes
    
    assert src.shape[0]==dst.shape[0]
    assert src.shape[1]==dst.shape[1]
    assert src.shape[1]==3    

    P = src.copy()
    Q = dst.copy()

    # center at origin
    m1 = np.mean(P, 0) 
    m2 = np.mean(Q, 0)

    P -= m1
    Q -= m2

    norm1 = np.linalg.norm(P)
    norm2 = np.linalg.norm(Q)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    P /= norm1
    Q /= norm2
    
    # use the matlab implementation that allows to handle reflections    
    R, s = orthogonal_procrustes_modified(Q, P)   
    
    scale = s*norm2/norm1
    t = m2-np.dot(m1*scale, R.T)
    
    mean_dist = average_distance(apply_rigid_transform(src, R, t, scale), dst)
    
    return scale, R, t, mean_dist

def point_set_registration(src, dst, fixed_scale=None, verbose=True):
    from scipy.optimize import minimize
    
    assert src.shape[0] == dst.shape[0]
    assert src.shape[1] == dst.shape[1]
    assert src.shape[1] == 3 
    
    def pack_params(R, t, scale):
        rvec = cv2.Rodrigues(R)[0]
        if fixed_scale is not None:
            return np.concatenate([rvec.ravel(), t, [fixed_scale]])
        else:
            return np.concatenate([rvec.ravel(), t, [scale]])
    
    def unpack_params(params):
        R, t, scale = cv2.Rodrigues(params[:3])[0], params[3:6], params[-1]
        if fixed_scale is not None:
            return R, t, fixed_scale
        else:
            return R, t, scale    

    _src, _dst = src.copy().astype(np.float32), dst.copy().astype(np.float32)
    
    if fixed_scale is not None:
        _, R, t, _ = procrustes_registration(_src*fixed_scale, _dst)
        scale = fixed_scale
    else:
        scale, R, t, _ = procrustes_registration(_src, _dst)
        
    mean_dist = average_distance(apply_rigid_transform(_src, R, t, scale), _dst)
    
    if verbose:
        logging.info("Initial guess using Procrustes registration:")
        logging.info("\t Mean error distance: {:0.3f} [unit of destination (dst) point set]".format(mean_dist))
        
    if np.linalg.det(R)<0:
        logging.info("!"*20)
        logging.info("Procrusted produced a rotation matrix with negative determinant.")
        logging.info("This implies that the coordinate systems of src and dst have different handedness.")
        logging.info("To fix this you have to flip one or more of the axis of your input..for example by negating them.")
        logging.info("!"*20)
    
    def funct(x):
        R, t, scale = unpack_params(x)
        src_transf = apply_rigid_transform(_src, R, t, scale)
        return average_distance(src_transf, _dst)  

    x0 = pack_params(R, t, scale)        
    res = minimize(funct, x0, method='Nelder-Mead', 
                   options={'maxiter':10000, 'disp':True}, 
                   tol=1e-24)
    #if verbose:
    #    logging.info(res)   
    
    R, t, scale = unpack_params(res.x)
    mean_dist = average_distance(apply_rigid_transform(_src, R, t, scale), _dst) 

    if verbose:
        logging.info("Final registration:")
        logging.info("\t Mean error distance: {:0.3f} [unit of destination (dst) point set]".format(mean_dist))     
    
    return scale, R, t, mean_dist