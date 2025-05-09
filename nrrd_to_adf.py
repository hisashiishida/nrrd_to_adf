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
import numpy as np
from seg_nrrd_to_pngs import SegNrrdCoalescer
from volume_data_to_slices import *

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

    def load(self, nrrd_hdr):
        space_directions = nrrd_hdr['space directions']
        if space_directions.shape[0] == 4: # Segmented NRRD, take the last three rows
            space_directions = space_directions[1:4, :]
        self.resolution = np.linalg.norm(space_directions, axis=1)
        
        sizes = nrrd_hdr['sizes']
        if sizes.shape[0] == 4: # Seg NRRD, take the last three values
            sizes = sizes[1:4]

        self.sizes = sizes
        self.dimensions = self.resolution * self.sizes
        self.coordinate_representation = nrrd_hdr['space'] # Usually LPS or RAS

        if self.coordinate_representation.lower() != 'left-posterior-superior':
            print("INFO! NRRD NOT USING LPS CONVENTION")
        
        rotation_offset = Rotation.from_euler('xyz', [0., 0., 0.], degrees=True)
        if self.coordinate_representation.lower() == 'right-anterior-superior':
            # Perform 180 degree rotation
            rotation_offset = Rotation.from_euler('xyz', [0., 0., 180.], degrees=True)
        # Add others
        
        U, _, Vt = np.linalg.svd(space_directions.T)
        self.orientation_mat = rotation_offset.as_matrix() @ (U @ Vt)
        self.orientation_rpy = Rotation.from_matrix(self.orientation_mat).as_euler('xyz', degrees=False) #lower case 'xyz' is extrinsic, uppercase 'XYZ' is instrinsic
        
        self.origin = nrrd_hdr['space origin']
        self.origin = rotation_offset.as_matrix() @ self.origin
    

class ADFData:
    def __init__(self):
        self.meta_data = OrderedDict()
        self.meta_data["ADF Version"] = 1.0
        self.meta_data["volumes"] = []
        self.meta_data["bodies"] = []

        self.volume_data = OrderedDict()
        self.volume_data["name"] = ""
        self.volume_data["location"] = OrderedDict()
        self.volume_data["location"]["position"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.volume_data["location"]["orientation"] = {"r": 0.0, "p": 0.0, "y": 0.0}
        self.volume_data["scale"] = 1.0
        self.volume_data["dimensions"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.volume_data["images"] = {"path": "", "prefix": "", "count": 0, "format": "png"}
        self.volume_data["iso-surface value"]= 0.5

        self.parent_body_data = OrderedDict()
        self.parent_body_data["name"] = ""
        self.parent_body_data["location"] = OrderedDict()
        self.parent_body_data["location"]["position"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.parent_body_data["location"]["orientation"] = {"r": 0.0, "p": 0.0, "y": 0.0}
        self.parent_body_data["mass"] = 0.0

    def set_volume_name_from_nrrd_filepath(self, nrrd_filepath):
        self.volume_data["volume filepath"] = nrrd_filepath
        self.set_volume_name(os.path.basename(nrrd_filepath).split('.')[0])

    def set_volume_name(self, name):
        self.volume_data["name"] = self.get_valid_ros_name(name)

    def set_volume_geometric_attributes(self, geometric_data: NrrdGeometricData):
        g = geometric_data
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

    def set_parent_body_name_attribute(self, name):
        self.parent_body_data["name"] = self.get_valid_ros_name(name)
        self.volume_data["parent"] = "BODY " + self.get_valid_ros_name(self.parent_body_data["name"])
    
    def set_parent_body_geometric_attributes(self, position, orientation):
        self.set_location_attributes(self.parent_body_data, position, orientation)

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

        return coalesced_data

    def save(self, adf_filepath):
        adf_data = self._coalesce_adf_data()
        # print("ADF Data\n", adf_data)
        setup_yaml()

        adf_folder = os.path.dirname(adf_filepath)
        if not os.path.exists(adf_folder):
                os.mkdir(adf_folder)

        with open(adf_filepath, 'w') as adf_file:
            yaml.dump(adf_data, adf_file, default_flow_style=False)
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
        valid_str = a_str.replace('-', '_')
        valid_str = valid_str.replace('.', '_')
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


def main():
    parser = ArgumentParser()
    parser.add_argument('-n', action='store', dest='nrrd_file', help='Specify NRRD filepath', required = True)
    parser.add_argument('-a', action='store', dest='adf_filepath', help='Specify ADF filepath', required = True)
    parser.add_argument('-p', action='store', dest='slices_prefix', help='Specify slices prefix', default='slice0')
    parser.add_argument('-s', action='store', dest="save_slices", help="Save slices. Can choose not to save slices again if they are already saved", default=True)
    parser.add_argument('--slices_path', action='store', dest="slices_path", help="Specify path for slices, defaults to the location of ADF filepath", default=None)
    
    parsed_args = parser.parse_args()
    print('Specified Arguments')
    print(parsed_args)

    nrrd_data, nrrd_hdr = nrrd.read(parsed_args.nrrd_file)

    nrrd_geometric_data = NrrdGeometricData()
    nrrd_geometric_data.load(nrrd_hdr)

    save_slices = False
    if parsed_args.save_slices in ['True', 'true', 'TRUE', 1, '1']:
        save_slices = True

    if not parsed_args.slices_path:
        parsed_args.slices_path = os.path.dirname(parsed_args.adf_filepath)
        print("INFO! Using the same path for slices as the ADF filepath")

    if save_slices:
        color_map = 'gray' if 'segmentation' not in nrrd_hdr.get('type', '').lower() else 'jet'
        if len(nrrd_data.shape) == 4:  # Handle 4D segmentation data
            # self.nrrd_data = np.sum(self.nrrd_data, axis=-1)  # Coalesce along the last dimension
            nrrd_coalescer = SegNrrdCoalescer()
            nrrd_coalescer.set_nrrd(nrrd_hdr, nrrd_data)
            nrrd_data = nrrd_coalescer.get_coalesced_data()

        save_volume_data_as_slices(nrrd_data, parsed_args.slices_path, parsed_args.slices_prefix, color_map)

    rel_slices_path = os.path.relpath(parsed_args.slices_path, os.path.dirname(parsed_args.adf_filepath))

    adf_data =  nrrd_to_adf(nrrd_geometric_data,
                            parsed_args.nrrd_file,
                            rel_slices_path,
                            parsed_args.slices_prefix)
    
    if len(nrrd_data.shape) == 4:
        seg_infos = SegNrrdCoalescer.get_segments_infos(nrrd_hdr)
        adf_data.meta_data["segments"] = OrderedDict()
        for seg_info in seg_infos:
            adf_data.meta_data["segments"][seg_info.index] = {"name": seg_info.name,
                                                              "color": seg_info.color.as_dict(),
                                                              "label": seg_info.label,
                                                              "index": seg_info.index}

    adf_data.save(parsed_args.adf_filepath)
    print("Exiting")
    

if __name__ == "__main__":
    main()