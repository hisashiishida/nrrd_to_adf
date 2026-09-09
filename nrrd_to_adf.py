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

from scipy.spatial.transform import Rotation
from collections import OrderedDict
from argparse import ArgumentParser
import nrrd
import yaml
import os
import shutil
from pathlib import Path
import numpy as np
from seg_nrrd_to_pngs import SegNrrdCoalescer
from volume_data_to_slices import *
import re
import json

class NrrdGeometricData:
    def __init__(self):
        self.origin = []
        self.orientation_rpy = []
        self.orientation_mat = None
        self.resolution = []
        self.dimensions = []
        self.sizes = []
        self.coordinate_representation = ""
        self.units_scale = 0.001 # NRRD is commonly in mm, convert to SI

        # Transform applied to the raw (i, j, k) voxel array so that every volume
        # ends up in the same anatomical voxel layout regardless of how Slicer
        # stored it (see _canonicalize / apply_canonical_orientation).
        self.canonical_axes_order = (0, 1, 2) # np.transpose order for the data array
        self.canonical_flip_axes = []         # data axes to np.flip after transposing

    def load(self, nrrd_hdr, canonical_orientation=True):
        space_directions = nrrd_hdr['space directions']
        if space_directions.shape[0] == 4: # Segmented NRRD, take the last three rows
            space_directions = space_directions[1:4, :]
        space_directions = np.array(space_directions, dtype=float)

        sizes = nrrd_hdr['sizes']
        if sizes.shape[0] == 4: # Seg NRRD, take the last three values
            sizes = sizes[1:4]
        sizes = np.array(sizes)

        self.coordinate_representation = nrrd_hdr['space'] # Usually LPS or RAS
        if self.coordinate_representation.lower() != 'left-posterior-superior':
            print("INFO! NRRD NOT USING LPS CONVENTION")

        origin = np.array(nrrd_hdr['space origin'], dtype=float)

        rotation_offset = Rotation.from_euler('xyz', [0., 0., 0.], degrees=True)
        if canonical_orientation:
            # Reorient the array (and its geometry) to a fixed anatomical voxel
            # layout. After this, space_directions is ~diag(+,-,-) * spacing for
            # EVERY input, so the SVD below yields the same orientation for all
            # volumes and the exported PNG stack is always consistent.
            space_directions, sizes, origin = self._canonicalize(
                space_directions, sizes, origin, self.coordinate_representation)
        else:
            # Legacy behaviour: no array reorientation, partial RAS handling only.
            if self.coordinate_representation.lower() == 'right-anterior-superior':
                # Perform 180 degree rotation
                rotation_offset = Rotation.from_euler('xyz', [0., 0., 180.], degrees=True)
            # Add others

        self.resolution = np.linalg.norm(space_directions, axis=1)
        self.sizes = sizes
        self.dimensions = self.resolution * self.sizes

        U, _, Vt = np.linalg.svd(space_directions.T)
        self.orientation_mat = rotation_offset.as_matrix() @ (U @ Vt)
        self.orientation_rpy = Rotation.from_matrix(self.orientation_mat).as_euler('xyz', degrees=False) #lower case 'xyz' is extrinsic, uppercase 'XYZ' is instrinsic

        self.origin = rotation_offset.as_matrix() @ origin

    def _canonicalize(self, space_directions, sizes, origin, space):
        """Return (space_directions, sizes, origin) reoriented so the voxel axes
        map to  i -> Left, j -> Anterior, k -> Inferior  (in LPS world space).

        This is the layout the AMBF multiview plugin and drill_location_in_volume
        already assume. As a side effect, records self.canonical_axes_order and
        self.canonical_flip_axes so the same permutation/flip can be applied to
        the voxel data before it is written out as PNG slices.
        """
        space = space.lower()
        if space in ('right-anterior-superior', 'ras'):
            # RAS -> LPS: negate the world X and Y components.
            ras_to_lps = np.array([-1.0, -1.0, 1.0])
            space_directions = space_directions * ras_to_lps
            origin = origin * ras_to_lps
        elif space not in ('left-posterior-superior', 'lps'):
            print(f"WARN! Unhandled NRRD space '{space}'. Assuming axes are already LPS-like.")

        # For each voxel axis, which world axis it mostly moves along, and the sign.
        perm = np.argmax(np.abs(space_directions), axis=1)
        signs = np.sign(space_directions[np.arange(3), perm])
        if sorted(int(p) for p in perm) != [0, 1, 2]:
            print(f"WARN! 'space directions' axis mapping {perm.tolist()} is not a clean "
                  "permutation (oblique volume). Slices may still be inconsistent; "
                  "consider resampling to an axis-aligned grid in 3D Slicer.")

        # 1) Transpose so that voxel axis n aligns with world axis n.
        axes_order = tuple(int(a) for a in np.argsort(perm))
        space_directions = space_directions[list(axes_order), :].astype(float)
        sizes = np.array(sizes)[list(axes_order)]
        signs = signs[list(axes_order)]

        # 2) Flip axes so the world-space signs become (+L, -P == Anterior,
        #    -S == Inferior), i.e. the "i=Left, j=Anterior, k=Inferior" layout.
        target_signs = np.array([1.0, -1.0, -1.0])
        flip_axes = [n for n in range(3) if signs[n] != target_signs[n]]
        for n in flip_axes:
            # New voxel 0 along axis n now sits at the opposite physical corner.
            # Shift by the full extent (sizes[n], not sizes[n]-1) so that the
            # downstream "origin + R @ (dimensions / 2)" volume-centre formula in
            # set_volume_geometric_attributes lands the box in the exact same
            # world location it would occupy without the flip.
            origin = origin + space_directions[n] * int(sizes[n])
            space_directions[n] = -space_directions[n]

        self.canonical_axes_order = axes_order
        self.canonical_flip_axes = flip_axes
        print(f"INFO! Canonical reorientation: transpose voxel axes {axes_order}, "
              f"flip axes {flip_axes}  (-> i=Left, j=Anterior, k=Inferior)")

        return space_directions, sizes, origin

    def apply_canonical_orientation(self, data):
        """Apply the permutation/flip found in load() to a voxel array.

        Handles a raw (i, j, k) array or a coalesced (i, j, k, 4) RGBA array.
        No-op when load() found the volume already canonical (e.g. PL04).
        """
        order = self.canonical_axes_order
        if order == (0, 1, 2) and not self.canonical_flip_axes:
            return data
        if data.ndim == 4:
            order = order + (3,)
        data = np.transpose(data, order)
        for n in self.canonical_flip_axes:
            data = np.flip(data, axis=n)
        return np.ascontiguousarray(data)


class ADFData:
    def __init__(self):
        self.meta_data = OrderedDict()
        self.meta_data["ADF Version"] = 1.0
        self.meta_data["volumes"] = []
        self.meta_data["bodies"] = []

        self.volume_data = {
            "name": "",
            "location": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"r": 0.0, "p": 0.0, "y": 0.0}
            },
            "scale": 1.0,
            "dimensions": {"x": 0.0, "y": 0.0, "z": 0.0},
            "images": {"path": "", "prefix": "", "count": 0, "format": "png"},
            "iso-surface value": 0.5,
        }
        self.volume_data["dimensions"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.volume_data["images"] = {"path": "", "prefix": "", "count": 0, "format": "png"}
        self.volume_data["iso-surface value"]= 0.5

        self.parent_body_data = {
            "name": "",
            "location": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"r": 0.0, "p": 0.0, "y": 0.0}
            },
            "scale": 1.0,
            "mass": 0.0
        }

        self.fiducials_data = []

    def set_volume_name_from_nrrd_filepath(self, nrrd_filepath):
        self.volume_data["volume filepath"] = nrrd_filepath
        self.set_volume_name(os.path.basename(nrrd_filepath).split('.')[0])

    def set_volume_name(self, name):
        self.volume_data["name"] = self.get_valid_ros_name(name)

    def set_volume_geometric_attributes(self, geometric_data: NrrdGeometricData):
        g = geometric_data
        if (g.orientation_mat is None):
            g.orientation_mat = Rotation.from_euler('xyz', g.orientation_rpy, degrees=False).as_matrix() #lower case 'xyz' is extrinsic, uppercase 'XYZ' is instrinsic
        origin = (g.origin + (g.orientation_mat @ (g.dimensions * 0.5))) * g.units_scale # AMBF takes middle as origin, so add rotated half dimensional offset
        dimensions = g.dimensions * g.units_scale
        self.set_location_attributes(self.volume_data, origin, g.orientation_rpy)
        
        self.volume_data["dimensions"]["x"] = float(dimensions[0])
        self.volume_data["dimensions"]["y"] = float(dimensions[1])
        self.volume_data["dimensions"]["z"] = float(dimensions[2])

    def set_volume_data_info_attributes(self, image_path, image_prefix, image_count, image_format):
        self.volume_data["images"]["path"] = image_path
        self.volume_data["images"]["prefix"] = image_prefix
        self.volume_data["images"]["count"] = int(image_count)
        self.volume_data["images"]["format"] = image_format

    def set_volume_shader_data(self, basepath, vs_filepath, fs_filepath):
        self.volume_data["shaders"] = OrderedDict()
        self.volume_data["shaders"]["path"] = basepath
        self.volume_data["shaders"]["vertex"] = vs_filepath
        self.volume_data["shaders"]["fragment"] = fs_filepath

    def set_volume_color_lut_data(self, lut_filepath):
        self.volume_data["color lut"] = lut_filepath

    def set_parent_body_name_attribute(self, name):
        self.parent_body_data["name"] = self.get_valid_ros_name(name)
        self.volume_data["parent"] = "BODY " + self.get_valid_ros_name(self.parent_body_data["name"])
    
    def set_parent_body_geometric_attributes(self, position, orientation):
        self.set_location_attributes(self.parent_body_data, position, orientation)

    def set_fiducials_data(self, fiducials_data_list):
        for data in fiducials_data_list:
            # Initialize fiducial data with default values, then set name and location attributes
            fiducial_data = {
                "name": data['name'],
                    "location": {
                    "position": {"x": data['position'][0], "y": data['position'][1], "z": data['position'][2]},
                    "orientation": {"r": 0.0, "p": 0.0, "y": 0.0}
                },
                "scale": 1.0,
                "mass": 0.0
            }
            self.fiducials_data.append(fiducial_data)

    def _coalesce_adf_data(self):
        coalesced_data = OrderedDict()
        coalesced_data = self.meta_data
        if self.volume_data["name"]:
            volume_identifier = "VOLUME " + self.volume_data["name"]
            coalesced_data["volumes"].append(volume_identifier)
            coalesced_data[volume_identifier] = self.volume_data
        if self.parent_body_data["name"]:
            body_identifier = "BODY " + self.parent_body_data["name"]
            coalesced_data["bodies"].append(body_identifier)
            coalesced_data[body_identifier] = self.parent_body_data

        if len(self.fiducials_data) > 0:
            for fiducial in self.fiducials_data:
                fiducial_identifier = "BODY " + fiducial["name"]
                # check if there is existing rigidbody with the same name
                if (fiducial_identifier not in coalesced_data["bodies"]):
                    coalesced_data["bodies"].append(fiducial_identifier)
                coalesced_data[fiducial_identifier] = fiducial

        return coalesced_data

    def save(self, adf_filepath):
        adf_data = self._coalesce_adf_data()
        # print("ADF Data\n", adf_data)
        setup_yaml()

        adf_folder = os.path.dirname(adf_filepath)
        if not os.path.exists(adf_folder):
                os.mkdir(adf_folder)

        with open(adf_filepath, 'w') as adf_file:
            yaml.dump(adf_data, adf_file, sort_keys=False, default_flow_style=False)
            print("Saving ADF", adf_filepath)
            adf_file.close()

    @staticmethod
    def set_location_attributes(yaml_data, position, orientation):
        yaml_data["location"]["position"]["x"] = float(position[0])
        yaml_data["location"]["position"]["y"] = float(position[1])
        yaml_data["location"]["position"]["z"] = float(position[2])

        yaml_data["location"]["orientation"]["r"] = float(orientation[0])
        yaml_data["location"]["orientation"]["p"] = float(orientation[1])
        yaml_data["location"]["orientation"]["y"] = float(orientation[2])

    @staticmethod
    def get_valid_ros_name(a_str: str):
        valid_str = re.sub(r'[^a-zA-Z0-9_/]', '', a_str)
        return valid_str


def represent_dictionary_order(self, dict_data):
        return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())

def setup_yaml():
    yaml.add_representer(OrderedDict, represent_dictionary_order)


def nrrd_to_adf(nrrd_geometric_data: NrrdGeometricData, nrrd_filepath="", slices_path="", slices_prefix=""):
    adf_data = ADFData()
    adf_data.set_volume_geometric_attributes(nrrd_geometric_data)
    adf_data.set_volume_name_from_nrrd_filepath(nrrd_filepath)
    adf_data.set_volume_data_info_attributes(slices_path, slices_prefix, nrrd_geometric_data.sizes[2], "png")
    adf_data.set_parent_body_name_attribute(adf_data.volume_data["name"] + "_Anatomical_Origin")
    return adf_data

def is_segmentation_file(filename: str):
    return filename.endswith('.seg.nrrd')


def load_fiducials(fiducial_filepath):
    fiducials_data = []
    with open(fiducial_filepath, 'r') as f:
        fiducials_json = json.load(f)["markups"][0]

        coordinate_system = fiducials_json["coordinateSystem"]
        coordinate_units = 1.0
        if fiducials_json["coordinateUnits"] == "mm":
            coordinate_units = 0.001

        for fiducial in fiducials_json["controlPoints"]:
            fiducial_name = fiducial["label"].replace("-", "_")  # ADF does not allow "-" , so replace with underscores
            if (coordinate_system == "LPS"):
                fiducials_data.append({
                    "name": fiducial_name,
                    "position": [fiducial["position"][0] * coordinate_units, fiducial["position"][1] * coordinate_units, fiducial["position"][2] * coordinate_units],
                    "orientation": fiducial["orientation"]
                })
            elif (coordinate_system == "RAS"):
                fiducials_data.append({
                    "name": fiducial_name,
                    "position": [-fiducial["position"][0] * coordinate_units, -fiducial["position"][1] * coordinate_units, fiducial["position"][2] * coordinate_units],
                    "orientation": fiducial["orientation"]
                })

    print(f"INFO! Loaded {len(fiducials_data)} fiducials from {fiducial_filepath}")
    return fiducials_data


def copy_shaders(from_path, to_path: str):
    shutil.copytree(
        Path(from_path),
        Path(to_path),
        dirs_exist_ok=True  # allows copying into existing directory
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('-n', '--nrrd_file', action='store', dest='nrrd_file', help='Specify NRRD filepath', required = True)
    parser.add_argument('-a', '--adf_filepath', action='store', dest='adf_filepath', help='Specify ADF filepath', required = True)
    parser.add_argument('-p', '--slices_prefix', action='store', dest='slices_prefix', help='Specify slices prefix', default='slice0')
    parser.add_argument('-c', '--color_lut', action='store', dest='color_lut', help='Set Color LUT', required=False)
    parser.add_argument('-s', '--save_slices', action='store', dest="save_slices", help="Save slices. Can choose not to save slices again if they are already saved", default=True)
    parser.add_argument('--slices_path', action='store', dest="slices_path", help="Specify path for slices, defaults to the location of ADF filepath", default=None)
    parser.add_argument('-f', '--fiducial_filepath', action='store', dest='fiducial_filepath', help='Specify fiducial JSON filepath (from 3D Slicer)', required=False, default=None)
    parser.add_argument('-v', '--volume_name', action='store', dest='volume_name', help='Override volume name (defaults to NRRD filename)', required=False, default=None)
    parser.add_argument('--canonical_orientation', action='store', dest='canonical_orientation', help="Reorient the volume to a fixed anatomical voxel layout (i=Left, j=Anterior, k=Inferior) so slices are consistent across NRRDs. Set to False for legacy behaviour.", default=True)

    parsed_args = parser.parse_args()
    print('Specified Arguments')
    print(parsed_args)

    nrrd_data, nrrd_hdr = nrrd.read(parsed_args.nrrd_file)

    _is_segmentation = is_segmentation_file(parsed_args.nrrd_file)

    canonical_orientation = parsed_args.canonical_orientation in [True, 'True', 'true', 'TRUE', 1, '1']

    nrrd_geometric_data = NrrdGeometricData()
    nrrd_geometric_data.load(nrrd_hdr, canonical_orientation=canonical_orientation)

    save_slices = False
    if parsed_args.save_slices in ['True', 'true', 'TRUE', 1, '1']:
        save_slices = True

    if not parsed_args.slices_path:
        parsed_args.slices_path = os.path.dirname(parsed_args.adf_filepath)
        print("INFO! Using the same path for slices as the ADF filepath")

    
    ### Begin Parsing ADF Data
    rel_slices_path = os.path.relpath(parsed_args.slices_path, os.path.dirname(parsed_args.adf_filepath))

    adf_data =  nrrd_to_adf(nrrd_geometric_data,
                            parsed_args.nrrd_file,
                            rel_slices_path,
                            parsed_args.slices_prefix)

    if parsed_args.volume_name:
        adf_data.set_volume_name(parsed_args.volume_name)
        adf_data.set_parent_body_name_attribute(adf_data.volume_data["name"] + "_Anatomical_Origin")

    if parsed_args.color_lut:
        print("INFO! Setting Color LUT to:", parsed_args.color_lut)
        rel_lut_path = os.path.relpath(parsed_args.color_lut, os.path.dirname(parsed_args.adf_filepath))
        adf_data.set_volume_color_lut_data(rel_lut_path)

    color_map = 'jet' if _is_segmentation else 'gray'

    if _is_segmentation:
        seg_infos = SegNrrdCoalescer.get_segments_infos(nrrd_hdr)
        adf_data.meta_data["segments"] = OrderedDict()
        for seg_info in seg_infos:
            adf_data.meta_data["segments"][seg_info.index] = {"name": seg_info.name,
                                                              "color": seg_info.color.as_dict(),
                                                              "label": seg_info.label,
                                                              "index": seg_info.index}

    ### Copy over shaders
    curr_filepath = os.path.abspath(__file__)
    if _is_segmentation:
        shader_from_dir = os.path.dirname(curr_filepath) + '/shaders/seg_nrrd'
    else:
        if parsed_args.color_lut:
            shader_from_dir = os.path.dirname(curr_filepath) + '/shaders/nrrd_lut'
        else:
            shader_from_dir = os.path.dirname(curr_filepath) + '/shaders/nrrd'

    shader_to_dir = os.path.dirname(parsed_args.adf_filepath) + '/shaders'
    copy_shaders(shader_from_dir, shader_to_dir)
    adf_data.set_volume_shader_data('shaders', 'shader.vs', 'shader.fs')

    ### Load Fiducials if provided
    if parsed_args.fiducial_filepath:
        if os.path.exists(parsed_args.fiducial_filepath):
            fiducials_data = load_fiducials(parsed_args.fiducial_filepath)
            if len(fiducials_data) > 0:
                adf_data.set_fiducials_data(fiducials_data)
        else:
            print(f"WARN! Fiducial filepath does not exist: {parsed_args.fiducial_filepath}")

    ### Save ADF Data as ADF File
    adf_data.save(parsed_args.adf_filepath)

    ### Save Slices
    if save_slices:
        if _is_segmentation:
            # self.nrrd_data = np.sum(self.nrrd_data, axis=-1)  # Coalesce along the last dimension
            nrrd_coalescer = SegNrrdCoalescer()
            nrrd_coalescer.parse_nrrd_data(nrrd_hdr, nrrd_data)
            nrrd_data = nrrd_coalescer.get_coalesced_data()

        # Reorient the voxel array to the same anatomical layout used for the ADF
        # geometry, so the PNG stack is consistent regardless of the source NRRD.
        nrrd_data = nrrd_geometric_data.apply_canonical_orientation(nrrd_data)

        save_volume_data_as_slices(nrrd_data, parsed_args.slices_path, parsed_args.slices_prefix, color_map)

    
    print("Exiting")
    

if __name__ == "__main__":
    main()
