#!/usr/bin/env python

import numpy as np
import os
import random
import sys

from PIL import Image, ImageEnhance, ImageOps

SIZE = [32, 64, 128]
COUNT = 20000

random.seed(0)


def generate(image, size):
    if random.randint(0, 1):
        image = ImageOps.mirror(image)

    ratio = random.uniform(0.8, 1.0) * size / max(image.size)
    image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.LANCZOS)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    result = Image.new('RGB', (size, size), (255, 255, 255))
    result.paste(image, ((size - image.size[0]) / 2, (size - image.size[1]) / 2), mask=image)
    return result


if __name__ == '__main__':
    images = []
    for imgpath in sys.argv[1:]:
        images.append(Image.open(imgpath))

    for size in SIZE:
        if not os.path.exists(str(size)):
            os.mkdir(str(size))
        for index in range(COUNT):
            image = generate(random.choice(images), size)
            image.save('%d/%05d.png' % (size, index))
