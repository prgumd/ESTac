###############################################################################
#
# Track a patch with LK like methods
#
# History:
# 11-15-23 - Levi Burner - Created file, LK like patch tracking using Jax
#
###############################################################################

import time
import cv2
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np

def I_W(I, yy_xx):
    I_W = jax.scipy.ndimage.map_coordinates(I, yy_xx, order=1, mode='nearest')
    return I_W

def I_W_p(I, p, W_params, W_p):
    yy_xx = W_p(p, *W_params)
    I_warped = I_W(I, yy_xx)
    return I_warped

def I_W_p_all(I, p, W_params_no_yy_xx, stride, p0, I_W_p, reshape=False):
    # Generate homogenous coordinates of points in image that will be sampled
    x = stride * np.arange(int(I.shape[1] / stride), dtype=np.float32)
    y = stride * np.arange(int(I.shape[0] / stride), dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    ones = np.ones_like(xx)
    yy_xx = np.dstack((yy, xx, ones)).reshape((x.shape[0]*y.shape[0], 3)).T

    W_params = (*W_params_no_yy_xx, yy_xx, p0)

    if len(I.shape) > 2:
        Is_warped = [I_W_p(I[:, :, i], p, W_params) for i in range(I.shape[2])]
        I_warped = jnp.dstack(Is_warped)
    else:
        I_warped = I_W_p(I, p, W_params)

    if reshape:
        return I_warped.reshape((y.shape[0], x.shape[0], I.shape[2]))
    else:
        return I_warped

def gauss_newton_step(I_warped_back_considered, T_considered, dT_dp, H_T_dp_inv):
    diff_I_T = I_warped_back_considered - T_considered
    grad_J_to_p = diff_I_T @ dT_dp
    delta_p_inv = H_T_dp_inv @ grad_J_to_p
    return delta_p_inv

def flow_loop_body(p_delta_p, I, W_params, GN_params, C_params, I_W_p, W_W_inv):
    p = p_delta_p[0:(p_delta_p.shape[0] - 1) // 2]
    steps = p_delta_p[-1]

    I_warped_back_considered = I_W_p(I, p, W_params)

    delta_p_inv = gauss_newton_step(I_warped_back_considered, *GN_params)
    p, delta_p = W_W_inv(p, delta_p_inv, *C_params)

    # print('p warped back')
    # print(p.reshape((2, 4)))
    # I_np = np.array(I_warped_back_considered).reshape((540, 960))
    # T_np = GN_params[0].reshape((540, 960))
    # diff_image = np.abs(I_np - T_np)
    # cv2.imshow('alignment', np.hstack((I_np, T_np, diff_image)))
    # cv2.waitKey(1)

    return jnp.array((*p, *delta_p, steps + 1))

def flow_loop(p, cond_parameters, body_parameters, cond, body):
    p_delta_p = jnp.concatenate((p, jnp.zeros(p.shape[0]+1,)))
    p_delta_p = body(p_delta_p, *body_parameters) # We want a do while loop
    p_delta_p_steps = jax.lax.while_loop(
                       lambda p_delta_p: cond(p_delta_p, *cond_parameters),
                       lambda p_delta_p: body(p_delta_p, *body_parameters),
                       p_delta_p)
    new_p = p_delta_p_steps[0:p.shape[0]]
    steps = p_delta_p_steps[2*p.shape[0]]
    return new_p, steps

def flow_loop_no_jit(p, cond_parameters, body_parameters, cond, body):
    p_delta_p = jnp.concatenate((p, jnp.zeros(p.shape[0]+1,)))
    p_delta_p = body(p_delta_p, *body_parameters) # We want a do while loop

    while cond(p_delta_p, *cond_parameters):
        p_delta_p = body(p_delta_p, *body_parameters)

    p_delta_p_steps = p_delta_p
    new_p = p_delta_p_steps[0:p.shape[0]]
    steps = p_delta_p_steps[2*p.shape[0]]
    return new_p, steps

def prepare_template(T, rect, R_fc_to_c, K, stride, p0, Sigma_inv, I_W_p, dI_W_p_dp):
    # Generate homogenous coordinates of points in template that will be sampled
    template_shape = jnp.array(((rect[3] - rect[1]) / stride,
                                (rect[2] - rect[0]) / stride), dtype=jnp.int32)
    x = rect[0] + stride * jnp.arange(template_shape[1], dtype=jnp.float32)
    y = rect[1] + stride * jnp.arange(template_shape[0], dtype=jnp.float32)
    xx, yy = jnp.meshgrid(x, y)
    ones = jnp.ones_like(xx)
    yy_xx = jnp.dstack((yy, xx, ones)).reshape((x.shape[0]*y.shape[0], 3)).T

    # Make a blurred template image, since that allows matching over great warp differences
    # jax_box = jnp.ones((9,9))
    jax_box = jnp.ones((5,5))
    jax_box /= jnp.sum(jax_box)
    T_blurred = jax.scipy.signal.convolve2d(T, jax_box, mode='same')
    # cv2.imshow('blurred', np.array(T_blurred))
    # cv2.waitKey(0)

    # Create approximate Hessian and Jacobian needed for Gauss Netwon updates
    K_inv = jnp.linalg.inv(K)
    W_params = (R_fc_to_c, K, K_inv, yy_xx, p0)
    T_blurred_considered = I_W_p    (T_blurred, p0, W_params)

    # cv2.imshow('T_blurred_considered', np.array(T_blurred_considered).reshape((80, 142)))

    dT_dp                = dI_W_p_dp(T_blurred, p0, W_params)
    H_T_dp = dT_dp.T @ dT_dp

    H_T_dp_inv = jnp.linalg.inv(H_T_dp + Sigma_inv)

    # Display grad images
    # def normalize(I):
    #     return (I - np.min(I) / (np.max(I)-np.min(I)))
    # cv2.imshow('grad images', np.vstack([normalize(dT_dp[:, i]).reshape(template_shape) for i in range(dT_dp.shape[1])]))
    # cv2.waitKey(0)
    return template_shape, yy_xx, T_blurred_considered, dT_dp, H_T_dp_inv

class JTrackRotInvariant:
    def __init__(self, template_image, rect, stride, R_c_fc, K,
                 delta_p_stop, max_steps,
                 p0, prepare_template, flow_loop, I_W_p,
                 blur_new_frame=False):
        self.K = K
        self.K_inv = jnp.linalg.inv(K)
        self.delta_p_stop = delta_p_stop
        self.max_steps = max_steps
        self.p0 = p0
        self.p = p0
        self.I_W_p = I_W_p
        self.flow_loop = flow_loop
        self.blur_new_frame = blur_new_frame
        (self.template_shape, self.yy_xx, self.T_blurred_considered,
         self.dT_dp, self.H_T_dp_inv) = prepare_template(template_image, rect, R_c_fc, K, stride)
        self.sec_it_jax = []
        # print('template shape', self.template_shape)

    def update(self, frame_gray, R_c_fc):
        if self.blur_new_frame:
            jax_box = jnp.ones((5,5))
            jax_box /= jnp.sum(jax_box)
            frame_gray = jax.scipy.signal.convolve2d(frame_gray, jax_box, mode='same')

        t_start = time.time()
        cond_parameters = (self.delta_p_stop, t_start, self.max_steps)

        W_params = (R_c_fc, self.K, self.K_inv, self.yy_xx, self.p0)
        GN_params = (self.T_blurred_considered, self.dT_dp, self.H_T_dp_inv)
        C_params = (self.p0,)
        body_parameters = (frame_gray, W_params, GN_params, C_params)

        self.p, steps = self.flow_loop(self.p, cond_parameters, body_parameters)
        t_end = time.time()

        sec_it = (t_end-t_start) / steps
        self.sec_it_jax.append(sec_it)

        if len(self.sec_it_jax) > 100: 
            self.sec_it_jax = self.sec_it_jax[-100:]
        else:
            pass
            #print('not trimmed')

        #print('Update time {} steps {:0.4f} seconds {:0.1f} Hz jax mean {:0.1f} std dev {:0.1f} jax'.format(steps, t_end - t_start, steps / (t_end-t_start), 1000*np.mean(self.sec_it_jax), 1000*np.std(self.sec_it_jax)))
        return self.p

    def visualize(self, frame_gray, R_c_fc):
        W_params = (R_c_fc, self.K, self.K_inv, self.yy_xx, self.p0)
        I_warped = self.I_W_p(frame_gray, self.p, W_params)
        return np.hstack((np.array(I_warped.reshape(self.template_shape)), np.array(self.T_blurred_considered.reshape(self.template_shape))))

def affine_W_p(p, R_c_fc, K, K_inv, yy_xx, p0):
    # Affine matrix parameterized by p
    A_p  = jnp.vstack((p.reshape((2, 3)), [0.0, 0.0, 1.0]))

    # Derived directly from perspective projection equations
    tmp = R_c_fc @ K_inv @ A_p

    K_cropped = K[0:2, :]
    top = K_cropped @ tmp
    bot = tmp[2, :]

    rot_times_affine = jnp.vstack((top, bot))

    # Swap X and Y axis on input and output
    # This the easiest way to accomdate R and intriniscs being in x,y order
    # while scipy coordinates are in y,x order
    S = jnp.array(((0., 1., 0.),
                   (1., 0., 0.),
                   (0., 0., 1.)))
    rot_times_affine_rolled = S @ rot_times_affine @ S

    new_yy_xx = rot_times_affine_rolled @ yy_xx

    final_yy_xx = (new_yy_xx[0:2, :] / new_yy_xx[2, :])
    return final_yy_xx

# Invert and compose the warps morally equivalent to eq 18 and eq 35 of LK 20 years on
def affine_compose_p_with_p_inv(last_p, delta_p_inv, p0):
    A_p_inv  = jnp.vstack(((p0 + delta_p_inv).reshape((2, 3)), [0.0, 0.0, 1.0]))
    A_last_p = jnp.vstack((last_p.reshape((2, 3)), [0.0, 0.0, 1.0]))

    A_p = jnp.linalg.inv(A_p_inv)
    A_new = A_last_p @ A_p

    delta_p = A_p  [0:2, :].flatten() - p0
    new_p   = A_new[0:2, :].flatten()
    return new_p, delta_p

def affine_flow_loop_cond(p_delta_p, delta_p_stop, start_t, max_steps):
    N_p = (p_delta_p.shape[0] - 1) // 2
    delta_p = p_delta_p[N_p:2*N_p]
    steps = p_delta_p[2*N_p]
    return (jnp.linalg.norm(delta_p) > delta_p_stop) & (steps < max_steps)

affine_p0 = jnp.array(((1.0, 0.0, 0.0),
                       (0.0, 1.0, 0.0))).flatten()
affine_I_W_p     = lambda *args: I_W_p(*args, affine_W_p)
affine_I_W_p_jit = jax.jit(affine_I_W_p)
affine_dI_W_p_dp = jax.jacfwd(affine_I_W_p, argnums=1)

affine_I_W_p_all = lambda I, p, R_c_fc, K, stride, reshape: I_W_p_all(I, p, (R_c_fc, K, jnp.linalg.inv(K)), stride, affine_p0, affine_I_W_p, reshape)
affine_I_W_p_all_jit = jax.jit(affine_I_W_p_all, static_argnames=('stride', 'reshape'))

affine_Sigma_inv = jnp.zeros((affine_p0.shape[0], affine_p0.shape[0]))
affine_prepare_template     = lambda *args: prepare_template(*args, affine_p0, affine_Sigma_inv, affine_I_W_p, affine_dI_W_p_dp)
affine_prepare_template_jit = jax.jit(affine_prepare_template, static_argnums=(1,))

affine_flow_loop_body = lambda *args: flow_loop_body(*args, affine_I_W_p, affine_compose_p_with_p_inv)
affine_flow_loop      = lambda *args: flow_loop     (*args, affine_flow_loop_cond, affine_flow_loop_body)
affine_flow_loop_jit = jax.jit(affine_flow_loop)

class JAffineTrackRotInvariant(JTrackRotInvariant):
    def __init__(self, **args):
        super().__init__(**args, p0=affine_p0, prepare_template=affine_prepare_template, I_W_p=affine_I_W_p_jit, flow_loop=affine_flow_loop_jit)

def homography_W_p(p, R_c_fc, K, K_inv, yy_xx, p0):
    # homography matrix parameterized by p
    # H_p = jnp.array((*p[0:6], 0.0, 0.0, 1.0)).reshape((3, 3))
    H_p = jnp.array((*p, 1.0)).reshape((3, 3))

    # Derived directly from perspective projection equations
    tmp = R_c_fc @ K_inv @ H_p

    K_cropped = K[0:2, :]
    top = K_cropped @ tmp
    bot = tmp[2, :]

    rot_times_homography = jnp.vstack((top, bot))

    # Swap X and Y axis on input and output
    # This the easiest way to accomdate R and intriniscs being in x,y order
    # while scipy coordinates are in y,x order
    S = jnp.array(((0., 1., 0.),
                   (1., 0., 0.),
                   (0., 0., 1.)))
    rot_times_homography_rolled = S @ rot_times_homography @ S

    new_yy_xx = rot_times_homography_rolled @ yy_xx

    final_yy_xx = (new_yy_xx[0:2, :] / new_yy_xx[2, :])
    return final_yy_xx

# Invert and compose the warps morally equivalent to eq 18 and eq 35 of LK 20 years on
def homography_compose_p_with_p_inv(last_p, delta_p_inv, p0):
    # print('last_p')
    # print(last_p)

    # print('delta_p_inv')
    # print(delta_p_inv)
    # delta_p_inv = jnp.array((*delta_p_inv[0:6], 0., 0.))

    H_p_inv = jnp.array((*(p0 + delta_p_inv), 1.0)).reshape((3, 3))
    H_last_p = jnp.array((*last_p, 1.0)).reshape((3, 3))

    H_p = jnp.linalg.inv(H_p_inv)
    H_new = H_last_p @ H_p

    delta_p = H_p.flatten()[0:8] - p0
    new_p   = H_new.flatten()[0:8]

    # print('new_p')
    # print(new_p)

    # print('delta_p')
    # print(delta_p)

    return new_p, delta_p

def homography_flow_loop_cond(p_delta_p, delta_p_stop, start_t, max_steps):
    N_p = (p_delta_p.shape[0] - 1) // 2
    delta_p = p_delta_p[N_p:2*N_p]
    steps = p_delta_p[2*N_p]
    return (jnp.linalg.norm(delta_p) > delta_p_stop) & (steps < max_steps)

homography_p0 = jnp.eye(3).flatten()[0:8]
homography_I_W_p     = lambda *args: I_W_p(*args, homography_W_p)
homography_I_W_p_jit = jax.jit(homography_I_W_p)
homography_dI_W_p_dp = jax.jacfwd(homography_I_W_p, argnums=1)

homography_I_W_p_all = lambda I, p, R_c_fc, K, stride, reshape: I_W_p_all(I, p, (R_c_fc, K, jnp.linalg.inv(K)), stride, homography_p0, homography_I_W_p, reshape)
homography_I_W_p_all_jit = jax.jit(homography_I_W_p_all, static_argnames=('stride', 'reshape'))

# homography_Sigma_inv = 0.01 * jnp.eye(homography_p0.shape[0])
homography_Sigma_inv = 0.05 * jnp.diag(jnp.array((0., 0., 1.,
                                                  0., 0., 1.,
                                                  0., 0.)))
homography_prepare_template     = lambda *args: prepare_template(*args, homography_p0, homography_Sigma_inv, homography_I_W_p, homography_dI_W_p_dp)
homography_prepare_template_jit = jax.jit(homography_prepare_template, static_argnums=(1,))

homography_flow_loop_body = lambda *args: flow_loop_body(*args, homography_I_W_p, homography_compose_p_with_p_inv)
homography_flow_loop      = lambda *args: flow_loop     (*args, homography_flow_loop_cond, homography_flow_loop_body)
homography_flow_loop_jit = jax.jit(homography_flow_loop)

class JHomographyTrackRotInvariant(JTrackRotInvariant):
    def __init__(self, **args):
        super().__init__(**args, p0=homography_p0, prepare_template=homography_prepare_template, I_W_p=homography_I_W_p_jit, flow_loop=homography_flow_loop_jit)


def matrix_cross(x):
    x_cross = jnp.array(((0., -x[2], x[1]),
                         (x[2], 0., -x[0]),
                         (-x[1], x[0], 0.)))
    return x_cross

# Defined as A x^T where x is column vector
# Useful for certain computer vision problems
# A x^T is defined as the matrix that results from
# interleaving the columns of A x_0, A x_1, ..., A x_n
# Useful for solving DLT like problems
def matrix_outer_vector(A, x):
    Ais = [A*x[i] for i in range(x.shape[0])]
    AxT = jnp.stack([Ais[i][:, j] for j in range(A.shape[1]) for i in range(x.shape[0])]).T
    return AxT

# Find homography such that
# u_j = H u_i
# By implementing a direct linear transform backed by eigh
# SVD is not used because jacobian estimation is not implemented
# The alternative, eigh, is not as numerically stable
def H_ji_4_point_dlt(u_i, u_j):
    # What follows needs to be done in float64 for numerical reasons
    # whether or not the SVD is used
    u_i = u_i.astype(jnp.float64)
    u_j = u_j.astype(jnp.float64) 

    # Convert to homogenous coordinates
    N_p = u_i.shape[0] // 2
    u_i = jnp.vstack((u_i.reshape((2, N_p)), jnp.ones((N_p,), dtype=u_i.dtype)))
    u_j = jnp.vstack((u_j.reshape((2, N_p)), jnp.ones((N_p,), dtype=u_j.dtype)))

    # Normalize to avoid numerical issues
    # norm_u_i = jnp.linalg.norm(u_i)
    u_i = u_i / jnp.linalg.norm(u_i)
    u_j = u_j / jnp.linalg.norm(u_j)

    # Take cross product of rhs with lhs so rhs is zero
    # rearrange lhs so that problem is 0 = A h where h = H.flatten()
    A_ks = []
    for k in range(N_p):
        u_i_k = u_i[:, k]
        u_j_k = u_j[:, k]
        u_j_k_x = matrix_cross(u_j_k)
        A_k = matrix_outer_vector(u_j_k_x, u_i_k)
        A_ks.append(A_k)
    A = jnp.vstack(A_ks)
    
    # sigma is sorted high to low
    # therefore bottom row of V is
    # h with norm 1 s.t. Ah is minimum (in least squares sense)
    # U, sigma, VT = jnp.linalg.svd(A)
    # H_ji = VT[-1, :].reshape((3,3))

    # Jacobian estimation is not implemented for SVD, so use eigh
    # here, the column of v corresponding to the minimum magnitude eigenvalue is H
    w, v = jnp.linalg.eigh(A.T @ A)
    w_min_i = jnp.argmin(jnp.abs(w))
    h = v[:, w_min_i]
    H_ji = h.reshape((3,3))

    return H_ji.astype(jnp.float32)

def hom_4p_W_p(p, R_c_fc, K, K_inv, yy_xx, p0):
    # Get 3x3 homography matrix transforming p0 to p
    H_p = H_ji_4_point_dlt(p0, p)

    # Derived directly from perspective projection equations
    tmp = R_c_fc @ K_inv @ H_p

    K_cropped = K[0:2, :]
    top = K_cropped @ tmp
    bot = tmp[2, :]

    rot_times_homography = jnp.vstack((top, bot))

    # Swap X and Y axis on input and output
    # This the easiest way to accomdate R and intriniscs being in x,y order
    # while scipy coordinates are in y,x order
    S = jnp.array(((0., 1., 0.),
                   (1., 0., 0.),
                   (0., 0., 1.)))
    rot_times_homography_rolled = S @ rot_times_homography @ S

    new_yy_xx = rot_times_homography_rolled @ yy_xx

    final_yy_xx = (new_yy_xx[0:2, :] / new_yy_xx[2, :])
    return final_yy_xx

# Invert and compose the warps morally equivalent to eq 18 and eq 35 of LK 20 years on
def hom_4p_compose_p_with_p_inv(last_p, delta_p_inv, p0):
    # print('last_p')
    # print(last_p.reshape((2, 4)))

    # print('p0')
    # print(p0.reshape((2, 4)))

    # print('delta_p_inv')
    # print(delta_p_inv.reshape((2, 4)))

    H_p_inv = H_ji_4_point_dlt(p0, p0 + delta_p_inv)
    H_last_p = H_ji_4_point_dlt(p0, last_p)

    H_p = jnp.linalg.inv(H_p_inv)
    H_new = H_last_p @ H_p

    # p0_norm = jnp.linalg.norm(p0)
    new_p = H_new[:, 0:2] @ (p0).reshape((2, 4)) + H_new[:, 3].reshape((3, 1))
    new_p = (new_p[0:2, :] / new_p[2, :]).flatten()
    delta_p = new_p.flatten() - last_p

    # print('new_p')
    # print(new_p.reshape((2, 4)))

    # print('delta_p')
    # print(delta_p.reshape((2, 4)))

    return new_p, delta_p

def hom_4p_flow_loop_cond(p_delta_p, delta_p_stop, start_t, max_steps):
    N_p = (p_delta_p.shape[0] - 1) // 2
    delta_p = p_delta_p[N_p:2*N_p]
    steps = p_delta_p[2*N_p]
    return (jnp.linalg.norm(delta_p) > delta_p_stop) & (steps < max_steps)

H_ji_4_point_dlt_jit = jax.jit(H_ji_4_point_dlt)

hom_4p_I_W_p     = lambda *args: I_W_p(*args, hom_4p_W_p)
hom_4p_I_W_p_jit = jax.jit(hom_4p_I_W_p)
hom_4p_dI_W_p_dp = jax.jacfwd(hom_4p_I_W_p, argnums=1)

hom_4p_I_W_p_all = lambda I, p, R_c_fc, K, stride, reshape, p0: I_W_p_all(I, p, (R_c_fc, K, jnp.linalg.inv(K)), stride, p0, hom_4p_I_W_p, reshape)
hom_4p_I_W_p_all_jit = jax.jit(hom_4p_I_W_p_all, static_argnames=('stride', 'reshape'))

hom_4p_Sigma_inv = 0.0 * jnp.eye(8)
# hom_4p_Sigma_inv = 0.05 * jnp.diag(jnp.array((0., 0., 1.,
#                                               0., 0., 1.,
#                                               0., 0.)))
hom_4p_prepare_template     = lambda *args: prepare_template(*args, hom_4p_Sigma_inv, hom_4p_I_W_p, hom_4p_dI_W_p_dp)
hom_4p_prepare_template_jit = jax.jit(hom_4p_prepare_template, static_argnums=(1,))

hom_4p_flow_loop_body = lambda *args: flow_loop_body(*args, hom_4p_I_W_p, hom_4p_compose_p_with_p_inv)
hom_4p_flow_loop      = lambda *args: flow_loop     (*args, hom_4p_flow_loop_cond, hom_4p_flow_loop_body)
# hom_4p_flow_loop      = lambda *args: flow_loop_no_jit(*args, hom_4p_flow_loop_cond, hom_4p_flow_loop_body)
hom_4p_flow_loop_jit = jax.jit(hom_4p_flow_loop)

class JHom4pTrackRotInvariant(JTrackRotInvariant):
    def __init__(self, **args):
        rect = args['rect']
        p0 = jnp.array(((rect[0], rect[2], rect[2], rect[0]),
                        (rect[1], rect[1], rect[3], rect[3])), dtype=jnp.float32).flatten()
        hom_4p_prepare_template_p0 = lambda *args: hom_4p_prepare_template(*args, p0)
        super().__init__(**args, p0=p0, prepare_template=hom_4p_prepare_template_p0, I_W_p=hom_4p_I_W_p_jit, flow_loop=hom_4p_flow_loop_jit)
        # super().__init__(**args, p0=p0, prepare_template=hom_4p_prepare_template_p0, I_W_p=hom_4p_I_W_p, flow_loop=hom_4p_flow_loop)



def test():
    # AxT = matrix_outer_vector(jnp.eye(3), jnp.array((1,2,3)))
    # print(AxT)

    # u_i = jnp.array(((0., 0.),
    #                  (1., 0.),
    #                  (1., 1.),
    #                  (0., 1.))).T.flatten()

    # u_j = jnp.array(((0., 0.),
    #                  (1., 0.),
    #                  (1., 1.),
    #                  (0., 1.))).T.flatten()


    # u_i = jnp.array([426., 854., 854., 426., 240., 240., 480., 480.]).astype(jnp.float32)
    # u_j = jnp.array([426., 854., 854., 426., 240., 240., 480., 480.]).astype(jnp.float32)

    u_i = jnp.array([426., 854., 854., 426., 240., 240., 480., 480.])
    # u_j = jnp.array([430., 854., 854., 426., 240., 240., 480., 480.])
    # u_j = u_i + 100
    u_j = jnp.array([425.94400024, 854.02520752, 853.5703125,  425.10858154, 239.24246216, 240.31121826, 477.90701294, 477.20950317])

    H_ji = H_ji_4_point_dlt(u_i, u_j)
    print(H_ji / H_ji[2, 2])

    # Check accuracy
    hat_u_j = H_ji[:, 0:2] @ u_i.reshape((2, 4)) + H_ji[:, 3].reshape((3, 1))
    hat_u_j = hat_u_j[0:2, :] / hat_u_j[2, :]

    print(u_j)
    print(hat_u_j.flatten())
    print(u_j - hat_u_j.flatten())
    print(u_j.dtype, H_ji.dtype, hat_u_j.dtype)

    # H_ji_4_point_dlt_jit = jax.jit(H_ji_4_point_dlt)
    # H_ji = H_ji_4_point_dlt_jit(u_i, u_j)
    # print(H_ji / H_ji[2, 2])

    # dH_ji_du_j_4_point_dlt_svd = jax.jacfwd(H_ji_4_point_dlt, argnums=1)
    # dH_ji_du_j = dH_ji_du_j_4_point_dlt_svd(u_i, u_j)
    # print(dH_ji_du_j.shape)

    # dH_ji_du_j_4_point_dlt_svd_jit = jax.jit(dH_ji_du_j_4_point_dlt_svd)
    # dH_ji_du_j = dH_ji_du_j_4_point_dlt_svd_jit(u_i, u_j)
    # print(dH_ji_du_j.shape)

if __name__ == '__main__':
    test()
