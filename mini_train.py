import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm

import data
import imlib as im
import DLlib as dl
import DMlib as dm

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../../OneDrive - Universidad Católica de Chile/Documents/datasets/'
dataset_hdf5 = 'INTA_GC_192_complex_2D.hdf5'
acqs, out_maps = data.load_hdf5(dataset_dir, dataset_hdf5, 12, end=231, MEBCRN=False)
acqs = acqs[:,::4,::4,:]
out_maps = out_maps[:,::4,::4,:]

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

trainX = acqs[:209,:,:,:]
trainY = out_maps[:209,:,:,:]

testX = acqs[222:223:,:,:,:]
testY = out_maps[222:223:,:,:,:]

# Overall dataset statistics
len_dataset,hgt,wdt,d_ech = np.shape(trainX)
_,_,_,n_out = np.shape(trainY)
ne = d_ech//2
r2_sc, fm_sc = 200.0, 300.0

print('Dataset size:', len_dataset)
print('Acquisition Dimensions:', hgt,wdt)
print('Output Maps:',n_out)

A_B_dataset = tf.data.Dataset.from_tensor_slices((trainX,trainY))
A_B_dataset = A_B_dataset.batch(1).shuffle(len_dataset)

# create a fixed beta schedule
timesteps = 200
beta = np.linspace(0.0001, 0.02, timesteps)

# this will be used as discussed in the reparameterization trick
alpha = 1 - beta
alpha_bar = np.cumprod(alpha, 0)
alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0)

# create our unet model
unet = dl.Unet(channels=6)

# create our checkopint manager
ckpt = tf.train.Checkpoint(unet=unet)
ckpt_manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=2)

# load from a previous checkpoint if it exists, else initialize the model from scratch
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    start_interation = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

# initialize the model in the memory of our GPU
test_images = np.ones([1, hgt, wdt, n_out])
test_timestamps = dm.generate_timestamp(0, 1, timesteps)
k = unet(test_images, test_timestamps)

# create our optimizer, we will use adam with a Learning rate of 1e-4
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)


def train_step(A, B):
    rng, tsrng = np.random.randint(0, 100000, size=(2,))
    timestep_values = dm.generate_timestamp(tsrng, B.shape[0], timesteps)

    noised_image, noise = dm.forward_noise(rng, B, timestep_values, alpha_bar)
    with tf.GradientTape() as tape:
        prediction = unet(noised_image, timestep_values)
        
        loss_value = dl.loss_fn(noise, prediction)
        # loss_value = dl.pi_loss_fn(noise, prediction, noised_image, A, timestep_values, alpha_bar)
    
    gradients = tape.gradient(loss_value, unet.trainable_variables)
    opt.apply_gradients(zip(gradients, unet.trainable_variables))

    return loss_value


epochs = 10
for e in range(1, epochs+1):
    # this is cool utility in Tensorflow that will create a nice looking progress bar
    bar = tf.keras.utils.Progbar(len_dataset-1)
    losses = []
    for i, (A, B) in enumerate(A_B_dataset):
        # run the training loop
        loss = train_step(A, B)
        losses.append(loss)
        bar.update(i, values=[("loss", loss)])

    avg = np.mean(losses)
    print(f"Average loss for epoch {e}/{epochs}: {avg}")
    ckpt_manager.save(checkpoint_number=e)


def abs_ch(x):
	return np.abs(tf.complex(x[:,:,:,0],x[:,:,:,1]))

def phi_ch(x):
    return x[:,:,:,1]

def r2s_ch(x):
    return x[:,:,:,0]


# Inference
x = tf.random.normal((1,hgt,wdt,n_out))
img_list = []
img_list.append(np.squeeze(abs_ch(x), 0))

for i in tqdm.tqdm(range(timesteps-1)):
    t = np.expand_dims(np.array(timesteps-i-1, np.int32), 0)
    pred_noise = unet(x, t)
    x = dm.ddpm(x, pred_noise, t, alpha, alpha_bar, beta) #- dm.ddpm_add_cond(x, testX, t, alpha, alpha_bar)
    img_list.append(np.squeeze(abs_ch(x), 0))

    if i % 25==0:
        plt.imshow(np.array(np.clip(np.squeeze(abs_ch(x), 0) * 255, 0, 255), np.uint8), cmap="gray")
        plt.show()

im.save_gif(img_list + ([img_list[-1]] * 100), "ddpm.gif", interval=20)

fig, axs = plt.subplots(figsize=(15, 8), nrows=2, ncols=3)
aux_im = np.abs(tf.squeeze(tf.complex(testX[:,:,:,0],testX[:,:,:,1])))
aux_w = np.abs(tf.squeeze(abs_ch(x)))
aux_f = np.abs(tf.squeeze(abs_ch(x[:,:,:,2:])))
aux_ff = aux_f/(aux_w+aux_f)
aux_ff[np.isnan(aux_ff)] = 0.0
aux_r2s = np.abs(tf.squeeze(r2s_ch(x[:,:,:,4:])))*r2_sc
aux_phi = np.abs(tf.squeeze(phi_ch(x[:,:,:,4:])))*fm_sc
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