import argparse
import cv2
import os
import time
import glob
import numpy as np

from openvino.inference_engine import IECore

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def load_to_IE(model_xml):
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    #reshape_input(net, pair)
    exec_net = ie.load_network(network=net, device_name="CPU")
    print("IR successfully loaded into Inference Engine.")
    del net
    return exec_net

def inference(args):
    '''
    Performs inference on an input image, given an ExecutableNetwork
    '''
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask)

    assert image.shape == mask.shape

    # byte alignment
    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)
    print(input_image.shape)

    exec_net = load_to_IE(args.model)
    # inference
    start = time.time()
    output_y = exec_net.infer({'input_x':input_image})
    print(time.time()-start)
    # restore to orignal resolution, cut off the padding
    result = output_y['inpaint_net/Tanh_1']
    cv2.imwrite(args.output, result)

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='examples/places2/case1_input.png', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--mask', default='examples/places2/case1_mask.png', type=str,
                        help='The filename of mask, value 255 indicates mask.')
    parser.add_argument('--output', default='output/case1.png', type=str,
                        help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='The directory of tensorflow checkpoint.')
    parser.add_argument('--model', default='model_ir_512x1360/deepfillv2_frozen.xml', type=str,
                        help='The directory of tensorflow checkpoint.')
    args = parser.parse_args()  

    '''
    conf = [{'model': '../models/model_ir_384x640/pwc_frozen.xml',
             'height': 384,
             'width': 640},
            {'model': '../models/model_ir_448x768/pwc_frozen.xml',
             'height': 448,
             'width': 768 },
            {'model': '../models/model_ir_640x832/pwc_frozen.xml',
             'height': 640,
             'width': 832 },
            {'model': '../models/model_ir_768x1024/pwc_frozen.xml',
             'height': 768,
             'width': 1024 },
            {'model': '../models/model_ir_768x1280/pwc_frozen.xml',
             'height': 768,
             'width': 1280 },
           ]

    opt = 0
    args.model = conf[opt]['model']
    args.height = conf[opt]['height']
    args.width = conf[opt]['width']
    '''
    
    return args

def main():
    args = get_args()
    inference(args)

if __name__ == "__main__":
    main()