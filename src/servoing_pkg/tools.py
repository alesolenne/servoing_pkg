import numpy as np
from pytransform3d.rotations import matrix_from_quaternion

def skew_matrix(v):
    """Create the skew matrix from a given vector.

    Parameters
    ----------
    v : [3x1] np.ndarray
        vector
            
    Returns
    -------
    skv_matr : [3x3] np.ndarray
               skew matrix of the given vector v
    """
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    skv_matr = skv - skv.T
    return(skv_matr)

def hom_matrix(t,q):
    """Create the homogeneus matrix transformation given translation vector t and quaternion q.

    Parameters
    ----------
    t : [3x1] np.ndarray
        translation vector
    
    q : [4x1] np.ndarray
        quaternion [x y z w] where x y z are the imaginary part and w the real part
            
    Returns
    -------
    T : [4x4] np.ndarray
        homogeneus matrix of t and q.
    """
    T = np.identity(4)
    T[:3,3]=t
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    q_new = [w,x,y,z]   # different order of elements in the representation of the quaternion
    R = matrix_from_quaternion(q_new)
    T[0:3,0:3] = R
    return(T)