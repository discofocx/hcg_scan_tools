""" Linear sequence trans-coder,
    used mainly for generating animated GIFs from a camera rig """

import os
# import re
# import time
# import datetime
import subprocess
# import logging

import cv2
import numpy as np

__author__ = '__discofocx__'
__copyright__ = 'Copyright 2018, HCG Technologies'
__version__ = '0.1.2'
__email__ = 'gsorchin@gmail.com'
__status__ = 'alpha'

# Globals

gDEBUG = False
gDECODER = 'ffmpeg'


def is_image(l_file):
    extensions = ['.jpg', '.jpeg', '.png']
    for e in extensions:
        if e in l_file.lower():
            return True
        else:
            continue
    return False


def image_sorter(image_path):
    """
    Due to the fact that images may have very different or inconsistent naming conventions, we try to enforce an image
    sorter function, in this case we harvest the last-most whole number of the file name and use it as a key to sort
    the images.
    :param image_path:
    :return: int, key for the built-in sort function
    """
    path, image = image_path

    try:
        sorting_number = image.split('.')[0].split('_')[-1]
    except Exception as e:
        print(e)
    else:
        return int(sorting_number)


def build_data_list(l_root):
    """
    We build a tuple list containing all the desired images for the sequence.
    (path, image)
    :param l_root: str, The directory to search for images
    :return: list, The tuple list
    """
    if os.path.isdir(l_root):

        data = list()

        '''
        Instead of using os.walk now we are using os.listdir, so that we only get the files in the first level
        of the given directory.
        
        for path, subdir, files in os.walk(l_root):
            for file in files:
                if is_image(file):
                    data.append((path, file))
                else:
                    continue
        '''

        for file in os.listdir(l_root):
            if is_image(file):
                data.append((os.path.abspath(l_root), file))
            else:
                continue

        if not data:
            raise ValueError('Not enough files to generate a sequence')
        else:
            data.sort(key=image_sorter)  # Automatically try to sort images
            return data
    else:
        raise NotADirectoryError('Please select a valid directory')


def set_sequence_settings(default=True):
    """ Set desired processing settings for the whole sequence,
        values are pulled from the GUI, otherwise, default values are used."""

    seq_settings = {'size': None,
                    'crop': None,
                    'watermark': None,
                    'rename': None,
                    'duration': None}
    if default:
        seq_settings['size'] = 'original'
        seq_settings['crop'] = False
        seq_settings['watermark'] = False
    else:
        # TODO Pull values from GUI
        pass

    return seq_settings


def process_sequence(**manifest):
    data_for_encode = None

    # Check for processing mode, if Boom, reverse and append
    if manifest['mode'] == 'Boom':
        manifest['data'] = append_boom_frames(manifest['data'])

    for index, data in enumerate(manifest['data']):
        # Unpack the data tuple and build a full system path
        path, file = data
        file_to_load = os.path.join(path, file)

        # Load the image into memory as a np array
        im = imread_alpha(file_to_load)

        # Reshape image
        im = reshape_image(im, **manifest)

        # Start apply watermark
        w_action, w_path = manifest['watermark']

        # Watermark parameters
        w_size = 1 / 4  # TODO Find a way to expose this parameter
        w_offset = (10, 10)  # TODO Offset, expose this parameter
        w_corner = 'BR'  # TODO Corner, expose this parameter

        if w_action:
            wm = imread_alpha(w_path)
            wm = resize_shape(im, wm, w_size)
            a, b, c, d = move_shape_around(im, wm, w_offset, w_corner)
            w_boundaries = (a, b, c, d)
            wm = key_shape(im, wm, w_boundaries)

            im[a:c, b:d] = wm

            if gDEBUG:
                cv2.imshow('wm', wm)

        else:
            pass

        if gDEBUG:
            cv2.imshow('window', im)
            k = cv2.waitKey(1)
            if k == 27:
                break

        # Rename
        if not manifest['rename'] == 'Type new name ...':
            numbering = '_' + str(index + 1001)
            file = manifest['rename'] + numbering + '.jpg'

        # Create auto path
        path = path + '\\' + manifest['rename']
        if not os.path.isdir(path):
            os.mkdir(path)

        # Save the frame
        save_file = os.path.join(path, file)
        cv2.imwrite(save_file, im)

        # Save some data for the encode later
        if index == 0:
            data_for_encode = (path, file)

        # We yield the index and the length to calculate progress
        yield index, len(manifest['data'])

    if gDEBUG:
        cv2.destroyAllWindows()

    res = encode_file(data_for_encode, manifest['duration'])
    print('Encoding result: {res}'.format(res=res))


def encode_file(im_data, gif_length):
    im_dir, im_file = im_data
    palette_out = os.path.join(im_dir, 'palette.png')

    im_list = [im for _, _, f in os.walk(im_dir) for im in f if is_image(im)]
    im_list.sort()

    padding = len(str(len(im_list)))

    if padding <= 3:
        padding = 4
    else:
        pass

    # Encoding framerate, we grab it from our file properties
    frames = len(im_list)
    desired_duration = gif_length

    fps = round(frames / desired_duration, 2)
    print(fps)

    # From where do we load our image sequence?
    frame_sequence_in = os.path.join(im_dir, im_file)
    frame_sequence_in = frame_sequence_in.replace('_1001.jpg', '_%0{0}d.jpg'.format(padding))

    # To where do we write our new encoded gif?
    gif_out = os.path.join(im_dir, im_file.replace('_1001.jpg', '.gif'))

    # Start encoding operation, we will use FFMPEG under a python subprocess call
    print('Encoding...')

    # palette = subprocess.Popen(
    #     '{0} -start_number 1001 -i {1} -vf palettegen {2}'.format(gDECODER, frame_sequence_in, palette_out),
    #     stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # , shell=True)

    palette = subprocess.Popen([gDECODER, '-start_number', '1001', '-i', frame_sequence_in,
                                '-vf', 'palettegen', palette_out], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = palette.communicate()
    pexit = palette.returncode

    if not pexit:

        print('Successfully generated a palette')

        # ffmpeg = subprocess.Popen(
        #     '{0} -start_number {1} -f image2 -r {2} -i {3} -i {4} -lavfi "paletteuse" -y {5}'.format(gDECODER, 1001,
        #                                                                                              fps,
        #                                                                                              frame_sequence_in,
        #                                                                                              palette_out,
        #                                                                                              gif_out),
        #     stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # , shell=True)

        ffmpeg = subprocess.Popen([gDECODER, '-start_number', '1001', '-f', 'image2', '-r', str(fps),
                                   '-i', frame_sequence_in, '-i', palette_out, '-lavfi', "paletteuse",
                                   '-y', gif_out], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out, err = ffmpeg.communicate()
        exit = ffmpeg.returncode

        if not exit:
            display = subprocess.Popen(gif_out, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            dout, derr = display.communicate()
            dexit = display.returncode
            if not dexit:
                return True
            else:
                print(derr)
                return False

        else:
            print(err)
            return False
    else:
        print(perr)
        return False


def append_boom_frames(data):
    rwd_data = data[::-1]  # Reverse original data
    rwd_data = rwd_data[1:-1]  # Remove unnecessary frames

    return data + rwd_data


def reshape_image(buffer, **data):
    # Unpack the parameters that we care about
    size = data['size'].rstrip(' SZ')
    crop = data['crop']

    size = convert_to_float(size)

    if size is not 1.0:
        ny, nx = buffer.shape[0] * size, buffer.shape[1] * size
        nx, ny = int(nx), int(ny)
        dst = cv2.resize(buffer, (nx, ny), interpolation=cv2.INTER_AREA)
    else:
        dst = buffer

    if crop:
        if nx > ny:  # Landscape
            min = ny
            max = nx
            x1 = int(max / 2) - int(min / 2)
            y1 = 0
            x2 = x1 + min
            y2 = min
        else:  # Portrait
            min = nx
            max = ny
            y1 = int(max / 2) - int(min / 2)
            x1 = 0
            y2 = y1 + min
            x2 = min

        dst = dst[y1:y2, x1:x2]

    return dst


def apply_watermark(buffer, watermark):
    wm_size = 5

    w_mark = cv2.imread(watermark)
    h1, w1, _ = w_mark.shape
    h2, w2, _ = buffer.shape

    th, tw = h2 / wm_size, w2 / wm_size

    f = th / h1

    h1, w1 = int(h1 * f), int(w1 * f)

    w_mark = cv2.resize(w_mark, (w1, h1), interpolation=cv2.INTER_AREA)

    return w_mark


def convert_to_float(fraction):
    try:
        return float(fraction)
    except ValueError:
        num, den = fraction.split('/')

    return float(num) / float(den)


# --- Watermark Start --- #


def create_solid(size_x, size_y, color):
    solid = np.ones((size_y, size_x, 3), dtype='uint8') * color

    return solid


def imread_alpha(path):
    im = cv2.imread(path)  # The cv2.IMREAD_UNCHANGED flag flips the image to horizontal orientation

    try:
        (B, G, R, A) = cv2.split(im)

    except ValueError:
        (B, G, R) = cv2.split(im)
        A = np.ones((im.shape[0], im.shape[1]), dtype='uint8')

    finally:
        B = cv2.bitwise_and(B, B, mask=A)
        G = cv2.bitwise_and(G, G, mask=A)
        R = cv2.bitwise_and(R, R, mask=A)

    im = cv2.merge([B, G, R, A])

    return im


def resize_shape(back, fore, factor):
    # Get the largest values from the two shapes
    b_max = max(back.shape[:2])
    f_max = max(fore.shape[:2])
    print(b_max)
    print(f_max)

    # Estimate ideal size based on factor
    i_size = factor * b_max / 100

    # Compute the factor by which we need to scale the fore
    scale_factor = i_size * 100 / f_max

    # Resize fore shape
    h, w, _ = fore.shape
    h = int(h * scale_factor)
    w = int(w * scale_factor)

    fore = cv2.resize(fore, (w, h), interpolation=cv2.INTER_AREA)

    return fore


def move_shape_around(back, fore, offset, corner):
    """ TL=Top left, TR=Top right, BL=Bottom left, BR=Bottom right """

    rows_o, cols_o = offset

    b_rows, b_cols, _ = back.shape
    f_rows, f_cols, _ = fore.shape

    if corner == 'TL':
        start_rows = rows_o
        start_cols = cols_o

    if corner == 'TR':
        start_rows = rows_o
        start_cols = b_cols - f_cols - cols_o

    if corner == 'BL':
        start_rows = b_rows - f_rows - rows_o
        start_cols = cols_o

    if corner == 'BR':
        start_rows = b_rows - f_rows - rows_o
        start_cols = b_cols - f_cols - cols_o

    end_rows = start_rows + f_rows
    end_cols = start_cols + f_cols

    return start_rows, start_cols, end_rows, end_cols


def key_shape(back, fore, boundaries):
    # Unpack boundaries
    sr, sc, er, ec = boundaries

    # Draw a ROI
    roi = back[sr:er, sc:ec]

    # Get mask and inverse it
    _, _, _, mask = cv2.split(fore)
    # grey = cv2.cvtColor(fore, cv2.COLOR_BGR2GRAY) - 2nd option mask with thresh
    # _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY) - 2nd option mask with thresh
    # mask_inv = cv2.bitwise_not(mask) - Draw an inverse mask

    # img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv) - Apply inverse mask
    img2_fg = cv2.bitwise_and(fore, fore, mask=mask)

    return cv2.multiply(roi, img2_fg)

# --- Watermark End --- #

# Execution
# if __name__ == '__main__':
#     if gDEBUG:
#         root = 'D:\\dev\\python\\hcg_scan_tools\\test_scan'
#     else:
#         pass
#
#     images = build_data_dict(root)
#     settings = set_sequence_settings()
#     process_sequence(images, **settings)
