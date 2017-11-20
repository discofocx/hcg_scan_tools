# imports
import numpy as np
import cv2


# functions

def image_read(read_path, alpha=False):
    """
    Read an image from disk, uses cv2, alpha=True will try to read a fourth channel, if not found,
    it will create a solid channel
    :param read_path: str, path to file in disk
    :param alpha: bool, whether the file should be read with a fourth channel
    :return: ndarray, image in buffer as numeric array
    """
    if alpha:
        image_buffer = cv2.imread(read_path, cv2.IMREAD_UNCHANGED)

        try:
            b, g, r, a = cv2.split(image_buffer)
            print('Alpha channel found!')
        except ValueError:
            b, g, r = cv2.split(image_buffer)
            a = np.ones((image_buffer.shape[0], image_buffer.shape[1]), dtype='uint8')
            print('Had to stack an extra channel')
        finally:
            b = cv2.bitwise_and(b, b, mask=a)
            g = cv2.bitwise_and(g, g, mask=a)
            r = cv2.bitwise_and(r, r, mask=a)

        image_buffer = cv2.merge([b, g, r, a])
    else:
        image_buffer = cv2.imread(read_path)

    return image_buffer
