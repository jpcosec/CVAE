# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import argparse
import tensorflow as tf
from img_manager import DataManager

from layered_model import CVAE
from tensorflow.contrib.tensorboard.plugins import projector

#todo documentar
# from tensorflow.contrib.training.python.training import hparam



def train(sess,
          model,
          triplet_manager,
          saver,
          make_embedding=False):
    summary_writer = tf.summary.FileWriter(flags.log_file, sess.graph)
    print("guardando grafo en " + flags.log_file)


    if make_embedding:
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.sprite.image_path = "sprite.png"
        embedding.sprite.single_image_dim.extend([16, 16])
        embedding.metadata_path = 'metadata.tsv'

    n_samples = triplet_manager.sample_size

    print("Entrenando sobre %s samples"% str(n_samples))

    # reconstruct_check_images = triplet_manager.get_random_images(10)

    indices = list(range(n_samples))

    total_batch = n_samples // flags.batch_size
    step = 0
    print("%s batches de %s" % (total_batch, flags.batch_size))

    # Training cycle
    for epoch in range(flags.epoch_size):
        # Shuffle img_features indices
        random.shuffle(indices)

        avg_cost = 0.0

        # Loop over all batches
        for i in range(total_batch):  # total_batch):

            # Generate triplet img_features batch
            batch_indices = indices[flags.batch_size * i: flags.batch_size * (i + 1)]
            [images, pos, neg] = triplet_manager.get_batch(batch_indices)

            # Fit training using batch data
            reconstr_loss, latent_loss, zmean, triplet_loss, summary_str = model.partial_fit(sess, images, pos, neg,
                                                                                             step)
            if i % 20 == 1:
                print('{"metric": "reconstruc_loss", "value": %s}' % reconstr_loss)
                print('{"metric": "latent_loss", "value": %s}' % latent_loss)
                print('{"metric": "triplet_loss", "value": %s}' % triplet_loss)

                # Visualizaciones
                if make_embedding:  # Arreglar para usar un conjunto fijo (sprite de wea)
                    try:
                        embedding_var = tf.Variable(zmean, name='z_mean')
                        embedding.tensor_name = embedding_var.name
                        projector.visualize_embeddings(summary_writer, config)

                        labels = triplet_manager.get_labels(batch_indices)
                        with open('metadata.tsv', 'w') as f:
                            f.write("Index\tLabel\n")
                            for i in range(flags.batch_size):
                                f.write("%d\t%d\n" % (batch_indices[i], labels[i]))
                    except:
                        print("problemas en embedding")

            summary_writer.add_summary(summary_str, step)
            step += 1
            # print("batch pasado")
            # Save checkpoint
            # saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step=step)

            # Image reconstruction check
            # reconstruct_check(sess, model, reconstruct_check_images)
            # todo devolver estos 2
            # Disentangle check
            # disentangle_check(sess, model, triplet_manager)
            # Save checkpoint
        saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step=step)


"""
def reconstruct_check(sess, model, images):
    # Check img_features reconstruction
    x_reconstruct = model.reconstruct(sess, images)

    if not os.path.exists("reconstr_img"):
        os.mkdir("reconstr_img")

    for i in range(len(images)):
        org_img = images[i].reshape(64, 64)
        org_img = org_img.astype(np.float32)
        reconstr_img = x_reconstruct[i].reshape(64, 64)
        imsave("reconstr_img/org_{0}.png".format(i), org_img)
        imsave("reconstr_img/reconstr_{0}.png".format(i), reconstr_img)


def disentangle_check(sess, model, manager, save_original=False):
    img = manager.get_image(shape=1, scale=2, orientation=5)
    if save_original:
        imsave("original.png", img.reshape(64, 64).astype(np.float32))

    batch_xs = [img]
    z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
    z_sigma_sq = np.exp(z_log_sigma_sq)[0]

    # Print variance
    zss_str = ""
    for i, zss in enumerate(z_sigma_sq):
        str = "z{0}={1:.4f}".format(i, zss)
        zss_str += str + ", "
    print(zss_str)

    # Save disentangled images
    z_m = z_mean[0]
    n_z = 10

    if not os.path.exists("disentangle_img"):
        os.mkdir("disentangle_img")

    for target_z_index in range(n_z):
        for ri in range(n_z):
            value = -3.0 + (6.0 / 9.0) * ri
            z_mean2 = np.zeros((1, n_z))
            for i in range(n_z):
                if (i == target_z_index):
                    z_mean2[0][i] = value
                else:
                    z_mean2[0][i] = z_m[i]
            reconstr_img = model.generate(sess, z_mean2)
            rimg = reconstr_img[0].reshape(64, 64)
            imsave("disentangle_img/check_z{0}_{1}.png".format(target_z_index, ri), rimg)

"""


def map(sess,
        model,
        manager):
    n_samples = manager.sample_size

    print("Entrenando sobre %s samples", n_samples)
    indices = list(range(n_samples))
    total_batch = n_samples // flags.batch_size
    step = 0
    print("%s batches de %s" % (total_batch, flags.batch_size))

    for i in range(n_samples):
        # Generate img_features batch
        batch_indices = indices[flags.batch_size * i: flags.batch_size * (i + 1)]
        [images, pos, neg] = manager.get_batch(batch_indices)

        reconstr_loss, latent_loss, zmean, triplet_loss, summary_str = model.partial_fit(sess, images, pos, neg,
                                                                                         step)


def load_checkpoints(sess, new=True):
    """
    Abre el checkpoint anterior
    :param sess:
    :param new:
    :return:
    """
    saver = tf.train.Saver()
    if new:
        return saver

    checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(flags.checkpoint_dir):
            os.mkdir(flags.checkpoint_dir)
    return saver


def main_fn(flags):
    manager = DataManager(filename=flags.image_dir,
                          gt_dir=None)

    sess = tf.Session()

    model = CVAE(gamma=flags.gamma,
                 capacity_limit=flags.capacity_limit,
                 capacity_change_duration=flags.capacity_change_duration,
                 learning_rate=flags.learning_rate,
                 im_size=manager.get_im_size)

    sess.run(tf.global_variables_initializer())

    saver = load_checkpoints(sess)

    if flags.training:
        print("Training")
        # Train
        train(sess, model, manager, saver)
        # train(sess, model, imcrop, saver)
    else:
        print("Falta implementar")
        # reconstruct_check_images = manager.get_random_images(10)
        # Image reconstruction check
        # reconstruct_check(sess, model, reconstruct_check_images)
        # Disentangle check
        # disentangle_check(sess, model, manager)

"""
    Main de parser
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--epoch_size',
        help=' "epoch size"',
        type=int,
        default=100000
    )
    parser.add_argument(
        '--batch_size',
        help=' "batch size"',
        type=int,
        default=64
    )
    parser.add_argument(
        '--gamma',
        help=' "gamma param for latent loss"',
        type=float,
        default=100.0
    )
    parser.add_argument(
        '--capacity_limit',
        help='"encoding capacity limit param for latent loss"',
        type=float,
        default=20.0
    )
    parser.add_argument(
        '--capacity_change_duration',
        help=' "encoding capacity change duration"',
        type=int,
        default=100000
    )
    parser.add_argument(
        '--learning_rate',
        help=' "learning rate"',
        type=float,
        default=5e-4
    )
    parser.add_argument(
        '--checkpoint_dir',
        help=' "checkpoint directory"',
        default="./output"
    )
    parser.add_argument(
        '--log_file',
        help=' log file directory',
        default="./output/logs"
    )
    parser.add_argument(
        '--training',
        help=' "True for training from data"',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--embedding',
        help=' True for embedding data',
        type=bool
    )
    #todo implementar estos dos ultimos desde el directorio
    parser.add_argument(
        '--image_dir',
        help=' "img_features direction"',
        default="test_IMG.jpg"
    )
    parser.add_argument(
        '--gt_dir',
        help=' Ground truth direction',
        default="test_segm.mat"
    )

    parser.add_argument(
        '--mensaje',
        help=' "Mensaje"',
        default="Experimentos TSEGMVAE"
    )
    flags = parser.parse_args()
    print(flags.mensaje)
    # Run the training job
    # flags=hparam.HParams(**args.__dict__)
    main_fn(flags)
