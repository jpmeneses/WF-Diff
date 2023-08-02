import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm

import data
import imlib as im
import DLlib as dl
import DMlib as dm
import pylib as py
import wflib as wf

from utils import *

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir', default='DDPM-000')
py.arg('--n_timesteps', type=int, default=200)
py.arg('--n_filters', type=int, default=64)
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_ckpt', type=int, default=10)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0001)
args = py.args()

# output_dir
output_dir = py.join('output',args.experiment_dir)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

ech_idx = args.n_echoes * 2
r2_sc, fm_sc = 200.0, 300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'JGalgani_GC_192_complex_2D.hdf5'
out_maps_1 = data.load_hdf5(dataset_dir, dataset_hdf5_1)

dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
out_maps_2 = data.load_hdf5(dataset_dir, dataset_hdf5_2)

dataset_hdf5_3 = 'INTArest_GC_192_complex_2D.hdf5'
out_maps_3 = data.load_hdf5(dataset_dir, dataset_hdf5_3)

dataset_hdf5_4 = 'Volunteers_GC_192_complex_2D.hdf5'
out_maps_4 = data.load_hdf5(dataset_dir, dataset_hdf5_4)

dataset_hdf5_5 = 'Attilio_GC_192_complex_2D.hdf5'
out_maps_5 = data.load_hdf5(dataset_dir, dataset_hdf5_5)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

trainX  = np.concatenate((out_maps_1,out_maps_3,out_maps_4,out_maps_5),axis=0)
valX    = out_maps_2

# Overall dataset statistics
len_dataset,hgt,wdt,n_out = np.shape(trainX)

print('Image Dimensions:', hgt,wdt)
print('Num. Output Maps:',n_out)

X_dataset = tf.data.Dataset.from_tensor_slices(trainX)
X_dataset = A_B_dataset.batch(args.batch_size).shuffle(len_dataset)
X_dataset_val = tf.data.Dataset.from_tensor_slices(valX)
X_dataset_val.batch(1)

################################################################################
########################### DIFFUSION TIMESTEPS ################################
################################################################################

# create a fixed beta schedule
beta = np.linspace(0.0001, 0.02, args.n_timesteps)

# this will be used as discussed in the reparameterization trick
alpha = 1 - beta
alpha_bar = np.cumprod(alpha, 0)
alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0)

# create our unet model
unet = dl.Unet(dim=args.n_filters, channels=n_out)

# create our checkopint manager
ckpt = tf.train.Checkpoint(unet=unet)
ckpt_manager = tf.train.CheckpointManager(ckpt, py.join("./output",args.experiment_dir,"checkpoints") , max_to_keep=2) #"./checkpoints"

# load from a previous checkpoint if it exists, else initialize the model from scratch
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    start_interation = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

# initialize the model in the memory of our GPU
test_images = np.ones([1, hgt, wdt, n_out])
test_timestamps = dm.generate_timestamp(0, 1, args.n_timesteps)
k = unet(test_images, test_timestamps)

# create our optimizer, we will use adam with a Learning rate of 1e-4
opt = tf.keras.optimizers.Adam(learning_rate=args.lr)


def train_step(X):
    rng, tsrng = np.random.randint(0, 100000, size=(2,))
    timestep_values = dm.generate_timestamp(tsrng, X.shape[0], args.n_timesteps)

    noised_image, noise = dm.forward_noise(rng, X, timestep_values, alpha_bar)
    with tf.GradientTape() as tape:
        prediction = unet(noised_image, timestep_values)
        
        loss_value = dl.loss_fn(noise, prediction)
    
    gradients = tape.gradient(loss_value, unet.trainable_variables)
    opt.apply_gradients(zip(gradients, unet.trainable_variables))

    return {'Loss': loss_value}


def validation_step(X):
    for i in range(args.n_timesteps-1):
        t = np.expand_dims(np.array(args.n_timesteps-i-1, np.int32), 0)
        pred_noise = unet(X, t)
        X = dm.ddpm(X, pred_noise, t, alpha, alpha_bar, beta)
    return X


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))
val_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'validation'))

# sample
val_iter = cycle(A_B_dataset_val)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)
n_div = np.ceil(total_steps/len(valX))

# main loop
for ep in range(args.epochs):
    if ep < ep_cnt:
        continue

    # update epoch counter
    ep_cnt.assign_add(1)

    # train for an epoch
    for X in X_dataset:
        # ==============================================================================
        # =                             DATA AUGMENTATION                              =
        # ==============================================================================
        for i in range(X.shape[0]):
            X_i = tf.squeeze(X[i,:,:,:], axis=0)
            p = np.random.rand()
            if p <= 0.4:
                # Random 90 deg rotations
                X_i = tf.image.rot90(X_i,k=np.random.randint(3))

                # Random horizontal reflections
                X_i = tf.image.random_flip_left_right(X_i)

                # Random vertical reflections
                X_i = tf.image.random_flip_up_down(X_i)
            if i <= 0:
                X_da = tf.expand_dims(X_i,axis=0)
            else:
                X_da = tf.concat([X_da, tf.expand_dims(X_i, axis=0)],axis=0)
        # ==============================================================================

        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        loss_dict = train_step(X_da)

        # summary
        with train_summary_writer.as_default():
            tl.summary(lLoss_dict, step=opt.iterations, name='Losses')

    if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
        ckpt_manager.save(checkpoint_number=ep+1)

    # Validation inference
    x_T = tf.random.normal((1,hgt,wdt,n_out))
    x = validation_step(x_T)
    y = wf.IDEAL_model(x)

    fig, axs = plt.subplots(figsize=(15, 8), nrows=2, ncols=3)
    
    aux_im = np.abs(tf.squeeze(tf.complex(y[:,:,:,0],y[:,:,:,1])))
    aux_w = np.abs(tf.squeeze(tf.complex(x[:,:,:,0],x[:,:,:,1])))
    aux_f = np.abs(tf.squeeze(tf.complex(x[:,:,:,2],x[:,:,:,3])))
    aux_ff = aux_f/(aux_w+aux_f)
    aux_ff[np.isnan(aux_ff)] = 0.0
    aux_r2s = np.abs(tf.squeeze(x[:,:,:,4]))*r2_sc
    aux_phi = np.abs(tf.squeeze(x[:,:,:,5]))*fm_sc
    
    acq_ax= axs[0,0].imshow(aux_im, cmap='gist_earth', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(acq_ax, ax=axs[0,0])
    axs[0,0].axis('off')
    w_ax = axs[0,1].imshow(aux_w, cmap='bone', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(w_ax, ax=axs[0,1])
    axs[0,1].axis('off')
    f_ax = axs[0,2].imshow(aux_f, cmap='pink', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(f_ax, ax=axs[0,2])
    axs[0,2].axis('off')
    ff_ax = axs[1,0].imshow(aux_ff, cmap='jet', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(ff_ax, ax=axs[1,0])
    axs[1,0].axis('off')
    r2s_ax = axs[1,1].imshow(aux_r2s, cmap='copper', interpolation='none', vmin=0, vmax=r2_sc)
    fig.colorbar(r2s_ax, ax=axs[1,1])
    axs[1,1].axis('off')
    phi_ax = axs[1,2].imshow(aux_phi, cmap='twilight', interpolation='none', vmin=-fm_sc, vmax=fm_sc)
    fig.colorbar(phi_ax, ax=axs[1,2])
    axs[1,2].axis('off')
    plt.show()

    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    make_space_above(axs,topmargin=0.8)
    plt.savefig(py.join(sample_dir, 'ep-%03d.png' % (ep+1)), bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)

