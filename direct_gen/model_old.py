from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import _pickle as pkl
from ops import *
from utils import *
from scipy import interpolate

class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, sample_size = 64, output_shape=(13,44),
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 train_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_shape: (optional) The resolution in pixels of the images. [64]
            sample_size: number of samples to output = 64
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_shape = output_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.train_dir = train_dir
        self.build_model()
        

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_shape[0], self.output_shape[1], self.c_dim],
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_shape[0], self.output_shape[1], self.c_dim],
                                        name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.z_sum = tf.summary.histogram("z", self.z)

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D, self.D_logits  = self.discriminator(self.images, self.y, reuse=False)
            #import IPython; IPython.embed()
            self.sampler = self.sampler(self.z, self.y)
            #import IPython; IPython.embed()
            self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits = self.discriminator(self.images)

            self.sampler = self.sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        if config.dataset == 'mnist':
            data_X, data_y = self.load_mnist()
        else:
            data_X, data_y, Ymean, Ystd, ps_std = self.load_pspec()
        import IPython; IPython.embed()
        with tf.variable_scope('optim'):
            d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                              .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                              .minimize(self.g_loss, var_list=self.g_vars)
        init = tf.global_variables_initializer()
        init.run()

        self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, 
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        summary_writer = tf.summary.FileWriter(config.train_dir, sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        
        sample_images = data_X[0:self.sample_size]
        sample_labels = data_y[0:self.sample_size]
            
        counter = 1
        start_time = time.time()

        if self.load(self.train_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)

        for epoch in xrange(config.epoch):
            seed = np.random.randint(1000)
            np.random.seed(seed)
            np.random.shuffle(data_X)
            np.random.seed(seed)
            np.random.shuffle(data_y)
            if True:
                batch_idxs = min(len(data_X), config.train_size) // config.batch_size
            else:            
                data = glob(os.path.join("./data", config.dataset, "*.jpg"))
                batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                if True:
                    batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                else:
                    batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_shape, is_grayscale = self.is_grayscale) for batch_file in batch_files]
                    if (self.is_grayscale):
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                if True:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    _ = self.sess.run(batchnorm_updates_op,
                        feed_dict={ self.z: batch_z, self.y:batch_labels })
                    
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels})
                    errD_real = self.d_loss_real.eval({self.images: batch_images, self.y:batch_labels})
                    errG = self.g_loss.eval({self.z: batch_z, self.y:batch_labels})
                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)
                    
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.images: batch_images})
                    errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if True:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images, self.y:batch_labels}
                        )
                    else:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images}
                        )
                    save_images(samples, [8, 8],
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.train_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4
        else:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])            
            h1 = tf.concat([h1, y],1)
            
            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = tf.concat([h2, y],1)

            h3 = linear(h2, 1, 'd_h3_lin')
            
            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        if not self.y_dim:
            s, ss = self.output_shape
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
            ss2, ss4, ss8, ss16 = int(ss/2), int(ss/4), int(ss/8), int(ss/16)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*ss16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s16, ss16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
                [self.batch_size, s8, ss8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                [self.batch_size, s4, ss4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                [self.batch_size, s2, ss2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                [self.batch_size, s, ss, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
        else:
            s, ss = self.output_shape
            s2, s4 = int(s/2), int(s/4) 
            ss2, ss4 = int(ss/2), int(ss/4)

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*ss4, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, s4, ss4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, ss2, self.gf_dim * 2], name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, ss, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            
            s = self.output_shape
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin'),
                            [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)
        else:
            s, ss = self.output_shape
            s2, s4 = int(s/2), int(s/4)
            ss2, ss4 = int(ss/2), int(ss/4)

            # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y],1)

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
            h0 = tf.concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*ss4, 'g_h1_lin'), train=False))
            h1 = tf.reshape(h1, [self.batch_size, s4, ss4, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, ss2, self.gf_dim * 2], name='g_h2'), train=False))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, ss, self.c_dim], name='g_h3'))

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
        y = np.concatenate((trY, teY), axis=0)
        
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
        dataset = os.path.join("/data1/21cmFast/pspec_emulator/", self.dataset_name)
        if not dataset.endswith('npz'):
            dataset = dataset+'.npz'
        dic = np.load(dataset)
        data = dic['data']
        grid = dic['grid']
        k_out = dic['k']
        redshift_out = dic['Z']

        pspec_std = np.std(data)

        data /= pspec_std
        #import IPython; IPython.embed()
        gridmean = np.mean(grid, axis=0, keepdims=True)
        gridstd = np.std(grid, axis=0, keepdims=True)

        grid = (grid - gridmean)/gridstd
        #
        
        return data, grid, gridmean[0], gridstd[0], pspec_std
            
    def save(self, train_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_shape)
        train_dir = os.path.join(train_dir, model_dir)

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        self.saver.save(self.sess,
                        os.path.join(train_dir, model_name),
                        global_step=step)

    def load(self, train_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_shape)
        train_dir = os.path.join(train_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(train_dir, ckpt_name))
            return True
        else:
            return False
