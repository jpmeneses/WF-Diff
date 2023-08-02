import tensorflow as tf
import numpy as np

# Multipeak fat model
species = ["water", "fat"]
ns = len(species)

field = 1.5

f_p = np.array([ 0., -3.80, -3.40, -2.60, -1.94, -0.39, 0.60 ]) * 1E-6 * 42.58E6 * field
f_p = tf.convert_to_tensor(f_p,dtype=tf.complex64)
f_p = tf.expand_dims(f_p,0)

A_p = np.array([[1.0,0.0],[0.0,0.087],[0.0,0.693],[0.0,0.128],[0.0,0.004],[0.0,0.039],[0.0,0.048]])
A_p = tf.convert_to_tensor(A_p,dtype=tf.complex64)

r2_sc = 200.0   # HR:150 / GC:200
fm_sc = 300.0   # HR:300 / GC:400
rho_sc = 1.4


@tf.function
def gen_M(te,get_Mpinv=True,get_P0=False):
    ne = te.shape[1] # len(te)
    te = tf.cast(te,tf.complex64)
    te = tf.expand_dims(te,-1)

    M = tf.linalg.matmul(tf.math.exp(tf.tensordot(2j*np.pi*te,f_p,axes=1)),A_p) # shape: bs x ne x ns

    Q, R = tf.linalg.qr(M)
    if get_P0:
        P0 = tf.eye(ne,dtype=tf.complex64) - tf.linalg.matmul(Q, tf.transpose(Q,perm=[0,2,1],conjugate=True))
        P0 = 0.5 * (tf.transpose(P0,perm=[0,2,1],conjugate=True) + P0)

    # Pseudo-inverse
    if get_Mpinv:
        M_pinv = tf.linalg.solve(R, tf.transpose(Q,perm=[0,2,1],conjugate=True))

    if get_P0 and get_Mpinv:
        return M, P0, M_pinv
    elif get_Mpinv and not(get_P0):
        return M, M_pinv
    elif not(get_Mpinv) and not(get_P0):
        return M


@tf.function
def IDEAL_model(out_maps):
    n_batch,hgt,wdt,_ = out_maps.shape
    ne = 6
    
    te = np.arange(start=1.3e-3,stop=12*1e-3,step=2.1e-3)
    te = tf.expand_dims(tf.convert_to_tensor(te,dtype=tf.float32),0) # (1,ne)
    te_complex = tf.complex(0.0,te) # (1,ne)
    
    M = gen_M(te,get_Mpinv=False) # (ne,ns)
    M = tf.squeeze(M,axis=0)

    # Generate complex water/fat signals
    out_wf = out_maps[:,:,:,:4]
    real_rho = out_wf[:,:,:,0::2]
    imag_rho = out_wf[:,:,:,1::2]
    rho = tf.complex(real_rho,imag_rho) * rho_sc

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    rho_mtx = tf.transpose(tf.reshape(rho, [n_batch, num_voxel, ns]), perm=[0,2,1]) # (nb,ns,nv)

    r2s = out_maps[:,:,:,4]
    phi = out_maps[:,:,:,5]

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi)) * fm_sc #/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,-1) # (nb,nv,1)

    Wp = tf.math.exp(tf.linalg.matmul(xi_rav, +2*np.pi * te_complex)) # (nb,nv,ne)
    Wp = tf.transpose(Wp, perm=[0,2,1]) # (nb,ne,nv)

    # Matrix operations
    Mp = tf.linalg.matmul(M,rho_mtx) # (nb,ne,nv)
    Smtx = Wp * Mp # (nb,ne,nv)

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(tf.transpose(Smtx, perm=[0,2,1]),[n_batch,hgt,wdt,ne])
    
    Re_gt = tf.math.real(S_hat)
    Im_gt = tf.math.imag(S_hat)
    zero_fill = tf.zeros_like(Re_gt,dtype=tf.float32)
    re_stack = tf.stack([Re_gt,zero_fill],4)
    re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*ne])
    im_stack = tf.stack([zero_fill,Im_gt],4)
    im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*ne])
    res_gt = re_aux + im_aux
    
    return res_gt


