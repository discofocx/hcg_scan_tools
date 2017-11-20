# imports

# functions


def _resize_watermark(background, watermark, factor):
    """
    Resize the watermark using the desired scale ratio
    :param background:
    :param watermark:
    :param factor:
    :return: ndarray, the re-sized watermark
    """

    # Get the largest values from the two shapes
    b_max = max(background.shape[:2])
    w_max = max(watermark.shape[:2])

    # Estimate ideal size based on factor //deprecated
    #  i_size = (factor * b_max) / 100 // deprecated

    # Compute the factor by which we need to scale the watermark
    scale_factor = (factor * b_max) / w_max  # ((factor * b_max) / 100) * 100 / w_max // deprecated

    # Resize watermark shape
    h, w, _ = watermark.shape
    h = int(h * scale_factor)
    w = int(w * scale_factor)
    resized_watermark = cv2.resize(watermark, (w, h), interpolation=cv2.INTER_AREA)

    return resized_watermark


def create_watermark(background, watermark, corner='BL', factor=(1/6), padding=(10,10)):
    """
    Superposes a watermark image on a background image
    :param background: ndarray, background image, alpha channel required
    :param watermark: ndarray, desired watermark image, alpha channel required
    :param corner: str, corner in background image to position the watermark, TL, TR, BL, BR
    :param factor: float, watermark size ratio
    :param padding: tuple, in pixels, padding on each side of the watermark
    :return: ndarray, the merged (blended) ROI still to be applied to the final composite
    """
    # Offset from the edges of the background to the watermark
    res = _resize_watermark(background, watermark, factor)