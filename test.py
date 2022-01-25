import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

def tf_Summary(sess, outdir='./model_logs'):
    # frozen and summary
    frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["inpaint_net/Tanh_1"])
    tf.summary.FileWriter(outdir, graph=frozen)
    tf.io.write_graph(frozen, outdir, "deepfillv2_frozen.pb", as_text=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='examples/places2/case1_input.png', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--mask', default='examples/places2/case1_mask.png', type=str,
                        help='The filename of mask, value 255 indicates mask.')
    parser.add_argument('--output', default='output/case1.png', type=str,
                        help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default='model_logs/places2_256', type=str,
                        help='The directory of tensorflow checkpoint.')

    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()

    model = InpaintCAModel()


    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask)
    # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # place_holder
        #input_image = tf.constant(input_image, dtype=tf.float32, name='input_x')
        #output = model.build_server_graph(FLAGS, input_image)
        place_holder = tf.placeholder(dtype=tf.float32, shape=[1,512,1360,3], name='input_x')
        output = model.build_server_graph(FLAGS, place_holder)

        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
            #print(from_name)
        sess.run(assign_ops)

        # Run the adapted samples through the network
        print('Model loaded.')
        start = time.time()
        feed_dict = {place_holder: input_image}
        #result = sess.run(output)
        result = sess.run(output, feed_dict=feed_dict)
        print(time.time()-start)
        cv2.imwrite(args.output, result[0][:, :, ::-1])

        tf_Summary(sess)
