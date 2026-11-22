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
# */
# //==============================================================================
import os
import re
import glob
import numpy as np
import nrrd
import matplotlib.image as mpimg
from argparse import ArgumentParser

from seg_nrrd_to_pngs import SegNrrdCoalescer
from nrrd_to_adf import NrrdGeometricData


def slice_files_in_order(folder, prefix):
    """
    Return the `<prefix><index>.png` files in `folder`, ordered by index.

    Accepts both zero-padded (plane00009.png) and unpadded (plane009.png)
    indices. Raises if an index appears twice (e.g. a folder mixing both
    naming styles) or if indices are missing, since either would misplace
    slices in the rebuilt volume.
    """
    rx = re.compile(re.escape(prefix) + r'(\d+)\.png$')
    indexed = {}
    for f in glob.glob(os.path.join(folder, prefix + '*.png')):
        m = rx.match(os.path.basename(f))
        if not m:
            continue
        idx = int(m.group(1))
        if idx in indexed:
            raise ValueError("Slice index %d appears twice: %s and %s (mixed padded/unpadded names?)"
                             % (idx, os.path.basename(indexed[idx]), os.path.basename(f)))
        indexed[idx] = f
    if not indexed:
        return []
    missing = sorted(set(range(max(indexed) + 1)) - set(indexed))
    if missing:
        raise ValueError("Missing slice indices %s in %s" % (missing[:10], folder))
    return [indexed[i] for i in sorted(indexed)]


def load_pngs_as_volume(folder, prefix):
    """
    Read all PNG slices matching `<prefix>*.png` from `folder` and stack them
    into an (X, Y, Z, C) volume. Reverses the CCW 90 rotation that
    save_volume_data_as_slices applies on write (np.rot90 with k=1) by
    rotating each slice with k=-1.
    """
    pattern = os.path.join(folder, prefix + '*.png')
    files = slice_files_in_order(folder, prefix)
    if not files:
        raise FileNotFoundError("No PNGs found matching pattern: " + pattern)

    print("INFO! Found", len(files), "PNG slices in", folder)

    first = np.rot90(mpimg.imread(files[0]), k=-1)
    if first.ndim == 2:
        h, w = first.shape
        channels = 1
    else:
        h, w, channels = first.shape

    if channels >= 3:
        volume = np.zeros((h, w, len(files), channels), dtype=np.float32)
    else:
        volume = np.zeros((h, w, len(files)), dtype=np.float32)

    for i, f in enumerate(files):
        im = mpimg.imread(f)
        im = np.rot90(im, k=-1)
        if im.dtype == np.uint8:
            im = im.astype(np.float32) / 255.0
        if channels >= 3:
            volume[:, :, i, :] = im
        else:
            volume[:, :, i] = im

    return volume


BACKGROUND = -1


def colors_to_segments(volume, segments_infos, alpha_threshold=0.05):
    """
    Map an (X, Y, Z, C) RGB(A) volume back into an (X, Y, Z) map of segment
    positions (indices into `segments_infos`), with BACKGROUND (-1) for empty
    voxels.

    This classifies by *segment*, not by label value: in Slicer .seg.nrrd
    files with separate layers, different segments can share the same label
    value (e.g. box and stone both LabelValue=1 on layers 0 and 1), so a
    label value alone cannot tell them apart.
    """
    if volume.ndim == 3:
        # Grayscale slice stack: pixel intensity encodes the label value (after
        # the 0..1 scaling done by mpimg.imread). Only unambiguous if every
        # segment has a distinct label value.
        labels = [s.label for s in segments_infos]
        if len(set(labels)) != len(labels):
            raise ValueError(
                "Grayscale slices cannot be mapped back to segments because several "
                f"segments share a label value ({labels}). Export RGBA slices instead."
            )
        intensity = (volume * 255.0).round().astype(np.int32)
        segmap = np.full(intensity.shape, BACKGROUND, dtype=np.int32)
        for pos, seg in enumerate(segments_infos):
            segmap[intensity == seg.label] = pos
        return segmap

    h, w, d = volume.shape[:3]
    rgb = volume[..., :3].astype(np.float32)
    seg_colors = np.array([s.color.as_list()[:3] for s in segments_infos], dtype=np.float32)

    if volume.shape[-1] == 4:
        is_bg = volume[..., 3] < alpha_threshold
    else:
        is_bg = np.all(rgb < alpha_threshold, axis=-1)

    flat_rgb = rgb.reshape(-1, 3)
    # Squared distance from every voxel to every segment color: (N, K)
    dists = np.sum((flat_rgb[:, None, :] - seg_colors[None, :, :]) ** 2, axis=-1)
    segmap = np.argmin(dists, axis=-1).astype(np.int32).reshape(h, w, d)
    segmap[is_bg] = BACKGROUND
    return segmap


def segments_to_nrrd_data(segmap, original_data, segments_infos):
    """
    Write a segment-position map back into the layout the original NRRD used.
    Slicer .seg.nrrd files come in two flavors:
      - 3D (collapsed labelmaps): a single (X, Y, Z) array of label values
      - 4D (separate layers): (L, X, Y, Z) where each layer L holds one or
        more non-overlapping segments, encoded by their label values
    Each segment's voxels are written only into its own layer, with its own
    label value.
    """
    data_dims = original_data.ndim
    if data_dims not in (3, 4):
        raise ValueError("Unsupported NRRD data dimensionality: " + str(data_dims))
    new_data = np.zeros_like(original_data)
    for pos, seg in enumerate(segments_infos):
        if seg.label is None:
            continue
        mask = (segmap == pos)
        if data_dims == 3:
            new_data[mask] = seg.label
        else:
            if seg.layer is None:
                continue
            new_data[seg.layer][mask] = seg.label
    return new_data


def main():
    parser = ArgumentParser(
        description="Update a Slicer .seg.nrrd file with the contents of a set of PNG slices "
                    "previously exported by seg_nrrd_to_pngs.py (or modified by an external tool)."
                    "(Claude Generated)"
    )
    parser.add_argument('-n', dest='nrrd_file', required=True,
                        help='Input .seg.nrrd file (used for header, segment colors and dimensions)')
    parser.add_argument('-s', dest='slices_path', required=True,
                        help='Folder containing the PNG slices')
    parser.add_argument('-p', dest='slices_prefix', default='slice0',
                        help='PNG filename prefix (default: slice0)')
    parser.add_argument('-o', dest='output_file', required=True,
                        help='Output .seg.nrrd file')
    parser.add_argument('--alpha-threshold', dest='alpha_threshold', type=float, default=0.05,
                        help='Voxels with alpha (or max RGB) below this are background (default: 0.05)')
    parser.add_argument('--canonical_orientation', dest='canonical_orientation', default=True,
                        help='Must match the value used when the slices were exported with '
                             'nrrd_to_adf.py (default: True). The slices are then in the canonical '
                             '(i=Left, j=Anterior, k=Inferior) layout and are reoriented back to the '
                             'source NRRD axis order before writing. Set to False for slices exported '
                             'with --canonical_orientation False.')
    parsed_args = parser.parse_args()
    print('Specified Arguments')
    print(parsed_args)

    print("INFO! Reading NRRD file:", parsed_args.nrrd_file)
    nrrd_data, nrrd_hdr = nrrd.read(parsed_args.nrrd_file)
    print("INFO! NRRD data shape:", nrrd_data.shape, " dtype:", nrrd_data.dtype)

    segments_infos = SegNrrdCoalescer.get_segments_infos(nrrd_hdr)
    print("INFO! Found", len(segments_infos), "segments in header")
    for s in segments_infos:
        s.print_info()
        print('-------------------')

    volume = load_pngs_as_volume(parsed_args.slices_path, parsed_args.slices_prefix)
    print("INFO! Loaded PNG volume shape:", volume.shape)

    # nrrd_to_adf.py reorients the voxel array to a canonical anatomical layout
    # before writing slices. Recompute that same transform from the header and
    # undo it so the slices line up with the source NRRD's native axis order.
    canonical_orientation = parsed_args.canonical_orientation in [True, 'True', 'true', 'TRUE', 1, '1']
    geometric_data = NrrdGeometricData()
    geometric_data.load(nrrd_hdr, canonical_orientation=canonical_orientation)
    if canonical_orientation:
        volume = geometric_data.undo_canonical_orientation(volume)
        print("INFO! Volume shape after undoing canonical orientation:", volume.shape)

    expected_xyz = nrrd_data.shape if nrrd_data.ndim == 3 else nrrd_data.shape[1:]
    if tuple(volume.shape[:3]) != tuple(expected_xyz):
        raise ValueError(
            f"PNG volume shape {tuple(volume.shape[:3])} does not match NRRD spatial shape "
            f"{tuple(expected_xyz)}. Check that -n is the .seg.nrrd the slices were exported "
            "from and that --canonical_orientation matches the export setting."
        )

    print("INFO! Mapping PNG colors back to segments...")
    segmap = colors_to_segments(volume, segments_infos,
                                alpha_threshold=parsed_args.alpha_threshold)

    print("INFO! Writing segments into NRRD layout...")
    updated = segments_to_nrrd_data(segmap, nrrd_data, segments_infos)

    print("INFO! Writing updated NRRD to:", parsed_args.output_file)
    nrrd.write(parsed_args.output_file, updated, nrrd_hdr)
    print("INFO! Done.")


if __name__ == '__main__':
    main()
