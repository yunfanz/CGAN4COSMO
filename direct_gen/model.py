from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=16, output_height=64, output_width=64,
                 y_dim=None, z_dim=1, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', train_dir=None, sample_dir=None, data_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.data_dir = data_dir

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.train_dir = train_dir

        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.c_dim = self.data_X[0].shape[-1]
        elif True:
            self.data_X, self.data_y, self.Ymean, self.Ystd, self.ps_std = self.load_pspec()
            self.c_dim = 1
            self.y_dim = self.data_y.shape[1]
        else:
            self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
            imreadImg = imread(self.data[0]);
            if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator( self.y)

        self.sampler = self.sampler(self.y)

        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)


        #self.g_loss = tf.reduce_mean(
        #    sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))


        self.create_loss_terms()
        



        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def create_loss_terms(self):
        #Bournoulli MLP, for positive data
        #self.reconstr_loss = \
        #    -tf.reduce_sum(self.batch_flatten * tf.log(1e-10 + self.batch_reconstruct_flatten)
        #                   + (1-self.batch_flatten) * tf.log(1e-10 + 1 - self.batch_reconstruct_flatten), 1)
        #Gaussian MLP
        self.batch_flatten = tf.reshape(self.inputs, [self.batch_size, -1])
        self.batch_reconstruct_flatten = tf.reshape(self.G, [self.batch_size, -1])
        self.reconstr_loss = 0.5*tf.reduce_sum(tf.square(self.batch_flatten-self.batch_reconstruct_flatten)/1)
        self.g_loss = self.reconstr_loss
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.g_sum = merge_summary([self.G_sum, self.g_loss_sum])
        # L2
        #self.reconstr_loss = tf.reduce_mean(tf.square(self.batch_flatten-self.batch_reconstruct_flatten))
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        # self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean)
        #                                    - tf.exp(self.z_log_sigma_sq), 1)
        # # self.latent_loss = -0.5 * tf.reduce_sum(1 + tf.log(self.z_sigma_sq) - tf.square(self.z_mean)
        # #                                    - self.z_sigma_sq, 1)
        # self.vae_loss = tf.reduce_mean(self.reconstr_loss + self.latent_loss) / self.x_dim 
        # # average over batch and pixel
        # #import IPython; IPython.embed()
        # self.r_loss_sum = tf.scalar_summary("r_loss", self.reconstr_loss)
        # self.l_loss_sum = tf.scalar_summary(["l_loss"], self.latent_loss)
        # self.vae_loss_sum = tf.scalar_summary("vae_loss", self.vae_loss)

    def train(self, config):

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.writer = SummaryWriter(config.train_dir, self.sess.graph)

        if True:
            sample_inputs = self.data_X[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        
        save_images(sample_inputs, image_manifold_size(sample_inputs.shape[0]),
                                    './{}/real_samples.png'.format(config.sample_dir))
    
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.train_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            seed = np.random.randint(1000)
            np.random.seed(seed)
            np.random.shuffle(self.data_X)
            np.random.seed(seed)
            np.random.shuffle(self.data_y)
            if True:
                batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
            else:      
                self.data = glob(os.path.join(
                    "./data", config.dataset, self.input_fname_pattern))
                batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                if True:
                    batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                    #print(epoch, idx, batch_images.shape, batch_labels.shape)
                
                                        # Update D network
                   
                        
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.y:batch_labels, self.inputs: batch_images})
                    self.writer.add_summary(summary_str, counter)

                    errG = self.g_loss.eval({
                            self.y: batch_labels,
                            self.inputs: batch_images
                    })
               

                counter += 1
                if np.mod(counter, 10) == 1:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errG))

                if np.mod(counter, 100) == 1:
                    #print('S',sample_inputs.shape, self.inputs.get_shape())
                    samples, g_loss = self.sess.run(
                        [self.sampler, self.g_loss],
                        feed_dict={
                                self.inputs: sample_inputs,
                                self.y:sample_labels,
                        }
                    )
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample]  g_loss: %.8f" % (g_loss)) 


                if np.mod(counter, 500) == 2:
                    self.save(config.train_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])      
                h1 = concat([h1, y], 1)
                
                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')
                
                return tf.nn.sigmoid(h3), h3

    def generator(self, y):
        with tf.variable_scope("generator") as scope:
            if True:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.y_, self.h0_w, self.h0_b = linear(
                        y, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(
                        self.y_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(
                        h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                        h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                        h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                        h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(
                        self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                        linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                        [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(
                         deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
                #return tf.nn.tanh(
                #         deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def sampler(self, y):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if True:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                        linear(y, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
                        [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                        linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                        deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)
        
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)
        
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0
        
        return X/255.,y_vec
    def load_pspec(self):
        dataset = os.path.join(self.data_dir, self.dataset_name)
        if not dataset.endswith('npz'):
            dataset = dataset+'.npz'
        dic = np.load(dataset)
        data = dic['data']
        grid = dic['grid']
        k_out = dic['k']
        redshift_out = dic['Z']

        #data = np.log(data)


        pspec_std = np.std(data)

        data /= pspec_std
        #import IPython; IPython.embed()
        gridmean = np.mean(grid, axis=0, keepdims=True)
        gridstd = np.std(grid, axis=0, keepdims=True)

        grid = (grid - gridmean)/gridstd
        #
        
        return data[...,np.newaxis], grid, gridmean[0], gridstd[0], pspec_std

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
                self.dataset_name, self.batch_size,
                self.output_height, self.output_width)
            
    def save(self, train_dir, step):
        model_name = "DCGAN.model"
        train_dir = os.path.join(train_dir, self.model_dir)

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        self.saver.save(self.sess,
                        os.path.join(train_dir, model_name),
                        global_step=step)

    def load(self, train_dir):
        import re
        print(" [*] Reading checkpoints...")
        train_dir = os.path.join(train_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(train_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
