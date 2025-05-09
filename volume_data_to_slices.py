#!/usr/bin/env python
# //==============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2019-2025

#     All rights reserved.

#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions
#     are met:

#     * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.

#     * Neither the name of authors nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#     COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.

#     \author    <amunawa2@jh.edu>
#     \author    Adnan Munawar
#     \version   1.0
# */
# //==============================================================================
import matplotlib.pyplot as plt
import os
import numpy as np
import PIL.Image

def normalize_data(data):
	max = data.max()
	min = data.min()
	normalized_data = (data - min) / float(max - min)
	return normalized_data


def scale_data(data, scale):
	scaled_data = data * scale
	return scaled_data

def save_volume_data_as_slices(data, folder, prefix, colormap):
        if folder:
            print("INFO! Slices path provided as: ", folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            if data is not None:
                for i in range(data.shape[2]):
                    im_name = folder + '/' + prefix + str(i) + '.png'
                    im_data = np.rot90(data[:, :, i], k=1) # For AMBF we need this CCW 90 degree rotation
                    # print("Before", im_data.shape)
                    im_data = np.ascontiguousarray(im_data) # To avoid the bug in imsave in matplotlib imsave (https://stackoverflow.com/questions/78269316/matplotlib-imsave-error-ndarray-is-not-c-contiguous-but-it-is)
                    # print("After", im_data.shape)

                    # When the slices for the nrrd
                    if colormap == 'gray':
                        plt.imsave(im_name, im_data, cmap=colormap)

                    # When the slices for the seg.nrrd
                    else:
                        if im_data.dtype != np.uint8:
                            im_data = (255.0 * im_data)
                            im_data = np.rint(im_data)
                                                
                        # convert to PIL RBGA image
                        img = PIL.Image.fromarray(np.uint8(im_data))
                        img.save(im_name)