import numpy as np
import tensorflow as tf

def loss_fn(real, generated):
    loss = tf.math.reduce_mean((real - generated) ** 2)
    return loss


######################################################################################################
################################## PHYSICS-INFORMED OPERATIONS #######################################
######################################################################################################

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


def grad_xi(acqs, out_maps): # Must be same shape as out_maps
    n_batch,hgt,wdt,d_ech = acqs.shape
    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    ne = d_ech//2
    
    te = np.arange(start=1.3e-3,stop=12*1e-3,step=2.1e-3)
    te = tf.expand_dims(tf.convert_to_tensor(te,dtype=tf.float32),0) # (1,ne)
    te_complex = tf.complex(0.0,te) # (1,ne)

    # Re-format acqs (Wirtinger derivatives)
    # acqs = tf.complex(0.5*acqs[:,:,:,:,0],-0.5*acqs[:,:,:,:,1]) # (nb,ne,hgt,wdt)

    acqs = tf.complex(acqs[:,:,:,0::2],acqs[:,:,:,1::2]) # (nb,hgt,wdt,ne)
    Smtx = tf.reshape(acqs, [n_batch,num_voxel,ne]) # (nb,nv,ne)

    M, P0, M_pinv = gen_M(te, get_Mpinv=True, get_P0=True) # (ne,ns) | (ne,ne)
    P0 = tf.squeeze(P0,axis=0)

    tf.debugging.check_numerics(out_maps, message='B numerical error')
    r2s_pi = out_maps[:,:,:,1]
    phi = out_maps[:,:,:,0]

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s_pi) * fm_sc
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,-1) # (nb,nv,1)

    Wp = tf.math.exp(tf.linalg.matmul(xi_rav, +2*np.pi * te_complex)) # (nb,nv,ne)
    if not(tf.reduce_all(tf.math.is_finite(tf.abs(Wp)))):
        raise Exception("Only Inf-values in Wp ")
    Wp = tf.where(tf.math.is_inf(tf.abs(Wp)),0.0,Wp)
    Wm = tf.math.exp(tf.linalg.matmul(xi_rav, -2*np.pi * te_complex)) # (nb,nv,ne)
    Wm = tf.where(tf.math.is_inf(tf.abs(Wm)),0.0,Wm)
    if tf.reduce_sum(tf.abs(Wp)) == 0.0 or tf.reduce_sum(tf.abs(Wp)):
        raise Exception("Only zero-values in Wp and/or Wm matrices")

    # Xi gradient, considering Taylor approximation
    t_d = tf.squeeze(tf.linalg.diag(2*np.pi*te_complex)) # (1,ne) --> (ne,ne)
    Wmts = Wm * tf.linalg.matvec(t_d,Smtx) # (nb,nv,ne)
    WpPWmts = Wp * tf.linalg.matvec(P0,Wmts) # (nb,nv,ne)

    PWms = tf.linalg.matvec(P0, Wm*Smtx) # (nb,nv,ne)
    tWpPWms = tf.linalg.matvec(t_d, Wp*PWms) # (nb,nv,ne)

    dop_dxi = (tWpPWms - WpPWmts) * fm_sc # (nb,nv,ne)
    op = tf.expand_dims(Wp * tf.linalg.matvec(P0, Wm * Smtx),-2) # (nb,nv,1,ne)

    grad_res = 2 * tf.linalg.matvec(op,dop_dxi) # (nb,nv,1)
    grad_res = tf.reshape(tf.squeeze(grad_res,axis=-1),[n_batch,hgt,wdt]) # (nb,hgt,wdt)
    grad_res_r = +2*tf.math.real(tf.expand_dims(grad_res,axis=-1))
    grad_res_i = -2*tf.math.imag(tf.expand_dims(grad_res,axis=-1))
    grad_res = tf.concat([grad_res_r,grad_res_i],axis=-1) # (nb,hgt,wdt,2)

    tf.debugging.check_numerics(grad_res, message='grad_res numerical error')

    return tf.where(tf.math.is_nan(grad_res),0.0,grad_res)


def pi_loss_fn(real, generated, noised_image, condition, t, alpha_bar):
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    one_minus_sqrt_alpha_bar = np.sqrt(1-alpha_bar)
    reshaped_one_minus_sqrt_alpha_bar_t = np.reshape(np.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1))
    grad = grad_xi(condition, noised_image)
    loss = tf.math.reduce_mean((real - generated + grad*reshaped_one_minus_sqrt_alpha_bar_t) ** 2)
    return loss