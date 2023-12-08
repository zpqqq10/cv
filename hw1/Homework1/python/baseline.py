# adopted from mainBaseline.m   

import os
import numpy as np
from skimage import io
from datasets import load_datadir_re
from myPMS import myPMS

CURRENT_PATH = os.path.dirname(__file__)

def main():
    dataFormat = 'PNG'

    # Data names
    dataNameStack = ['bear', 'cat', 'pot', 'buddha']

    for testId, dataName in enumerate(dataNameStack):
        datadir = os.path.join(CURRENT_PATH, '..', 'pmsData', f'{dataName}{dataFormat}')
        # TODO cannot read 16-bit
        bitdepth = 8
        gamma = 1
        resize = 1
        data = load_datadir_re(datadir, bitdepth, resize, gamma)
        print('data loaded')

        rgb = (len(data['mask'].shape) == 3)

        if rgb:
            mask1 = np.mean(data['mask'], axis=-1)
            
        else:
            mask1 = data['mask']
            
        mask3 = np.repeat(mask1[:, :, np.newaxis], 3, axis=-1)
        # breakpoint()
        m = np.where(mask1 == 1)
        # p = len(m[0])

        # Standard photometric stereo
        Albedo, Normal, Render = myPMS(data, m)

        io.imsave(f'results/{dataName}_Albedo.png', (np.clip(Albedo[:, :, None] * 255 * 3, a_min=0, a_max=255.) * mask3).astype(np.uint8))
        # Save results as "png"
        # not using Normal * 255: consider negative values
        io.imsave(f'results/{dataName}_Normal.png', ((Normal + 1) * 128 * mask3).astype(np.uint8))
        io.imsave(f'results/{dataName}_Render.png', (np.clip(Render * 255 * 3, a_min=0, a_max=255.) * mask3).astype(np.uint8))

        # Save results as "mat"
        np.save(f'results/{dataName}_Normal.npy', Normal)
        print(f'{dataName} done')

if __name__ == "__main__":
    main()
