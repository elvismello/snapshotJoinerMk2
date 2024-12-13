#!/bin/python3

"""
DESCRIPTION:

usage: python snapshotJoiner.py haloA haloB haloAB  x y z  vx vy vz  rx ry rz

the last parameters are relative positions in x y z, relative
velocities in vx vy vz, then angles of rotation around the x, y and z axis,
respectively.


Script that joins snapshots.

"""


import numpy as np
import h5py
import argparse


def rotation (vector, alpha=0.0, beta=0.0, gamma=0.0, returnMatrix=False, dtype="float64"):
    """
    Expects radians.
    
    alpha: angle of rotation around the x axis
    beta: angle of rotation around the y axis
    gamma: angle of rotation around the z axis
    
    """
    # It may be better to find a way to apply a rotation without using an external for loop.
    vector = np.array(vector)

    #rotation matrix in x
    rAlpha = np.array([[1,             0,              0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]], dtype=dtype)
    
    #rotation matrix in y
    rBeta = np.array([[np.cos(beta),  0,  np.sin(beta)],
                      [0,             1,             0],
                      [-np.sin(beta), 0, np.cos(beta)]], dtype=dtype)

    #rotation matrix in z
    rGamma = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma),  np.cos(gamma), 0],
                       [0,                          0, 1]], dtype=dtype)

    rGeneral = np.matmul(np.matmul(rGamma, rBeta), rAlpha, dtype=dtype)
    

    if not returnMatrix:
        return np.matmul(rGeneral, np.array(vector), dtype=dtype)
    else:
        return rGeneral



def get_master_structure (snapshot_zero, snapshot_one):
    """
    Returns the output structure, keeping all datasets from the original
    snapshots.

    Ignores Header, Config and Parameters groups.
    
    """

    master_structure = {}
    for current_group in snapshot_zero:
        if current_group not in ["Header", "Config", "Parameters"]:
            master_structure[current_group] = [*snapshot_zero[current_group].keys()]

    for current_group in snapshot_one:
        if current_group not in ["Header", "Config", "Parameters"]:
            # has group
            if current_group in master_structure:
                # has group, but maybe not all datasets
                for current_dataset in snapshot_one[current_group]:
                    if not (current_dataset in master_structure[current_group]):
                        master_structure[current_group].append(current_dataset)
            # doesn't have the group
            else:
                master_structure[current_group] = [*snapshot_one[current_group].keys()]

    return master_structure



def solve_group (group_zero, group_one, group_master, rotation_angles,
                 relative_pos, relative_vel):
    """
    Solves how datasets should be joined, filling missing information with
    zeros.

    group_zero: first group that will NOT be rotated nor translated.
    group_one: second groups that WILL be rotated and translated.
    group_master: expected resulting structure.
    rotation_angles: angles to be used when applying rotation.
    relative_pos: relative position for translating the second snapshot.
    relative_vel: relative velocity for adding to the second snapshot.
    """


    solved_groups = {}
    for current_dataset in group_master:
        current_data = []
        if current_dataset in group_zero:
            current_data.append(group_zero[current_dataset][:])
        else:
            # fills missing information with zeros the size of the first
            # dataset of this group. A bit of gambiarra.
            current_data.append(np.zeros_like(group_zero["Masses"][:]))

        
        if current_dataset in group_one:
            data_ = group_one[current_dataset][:]

            # exception for position and velocity, which should be rotated and translated
            if current_dataset in ["Coordinates", "Velocities"]:
                if np.any(rotation_angles):
                    data_ = np.array([rotation(h, *rotation_angles) for h in data_])

                if current_dataset == "Coordinates":
                    print(current_dataset)
                    data_ = data_ + relative_pos

                if current_dataset == "Velocities":
                    data_ = data_ + relative_vel

                current_data.append(data_)
            
            else:
                current_data.append(group_one[current_dataset][:])
        else:
            # fills missing information with zeros the size of the Masses
            # dataset of this group. A bit of gambiarra, but all particles
            # should have mass.
            current_data.append(np.zeros_like(group_one["Masses"][:]))

        solved_groups[current_dataset] = np.concatenate(current_data)

    return solved_groups



def fix_ids_and_npart (all_data):
    """
    Resets all ids, reasigning values that make sense for the output snapshot.
    Also returns npart.
    """

    npart = []
    for current_group in all_data:
        if "ParticleIDs" in all_data[current_group]:
            npart.append(len(all_data[current_group]["ParticleIDs"]))
        else:
            npart.append(0)


    ids = []
    for i, j in enumerate(npart):
        sum = np.sum(npart[:i], dtype="uint32")
        ids.append(range(sum + 1, sum + j + 1))


    for current_group, current_ids in zip(all_data, ids):
        if "ParticleIDs" in all_data[current_group]:
            all_data[current_group]["ParticleIDs"] = np.array(current_ids,
                                                              dtype="uint32")
            
    return all_data, npart



def write_header(file, n_part, double_precision=1):
    #TODO enable funtionalities from the config parser
    #TODO make it less hardcoded
    header = file.create_group('Header')
    header.attrs['NumPart_ThisFile'] = np.asarray(n_part)
    header.attrs['NumPart_Total'] = np.asarray(n_part)
    header.attrs['NumPart_Total_HighWord'] = 0 * np.asarray(n_part)
    header.attrs['MassTable'] = np.zeros(6)
    header.attrs['Time'] = float(0.0)
    header.attrs['Redshift'] = float(0.0)
    header.attrs['BoxSize'] = float(0.0)
    header.attrs['NumFilesPerSnapshot'] = int(1)
    header.attrs['Omega0'] = float(0.0)
    header.attrs['OmegaLambda'] = float(0.0)
    header.attrs['HubbleParam'] = float(1.0)
    header.attrs['Flag_Sfr'] = int(0.0)
    header.attrs['Flag_Cooling'] = int(0)
    header.attrs['Flag_StellarAge'] = int(0)
    header.attrs['Flag_Metals'] = int(0)
    header.attrs['Flag_Feedback'] = int(0)
    header.attrs['Flag_DoublePrecision'] = double_precision
    header.attrs['Flag_IC_Info'] = 0



def write_snapshot (all_data, output_name, npart):
    """
    Writes all the data in hdf5 format.

    Assumes all_data is a dict of dicts, where the first key sets the group and
    the second sets the dataset containg arrays.

    """

    with h5py.File(output_name, "w") as output_snapshot:

        write_header(output_snapshot, npart)

        for current_group in all_data:
            group_object = output_snapshot.create_group(current_group)

            for current_dataset in all_data[current_group]:
                group_object.create_dataset(current_dataset,
                                data=all_data[current_group][current_dataset])



def join (snapshot_zero, snapshot_one, output="init.hdf5",
          relative_pos=[0.0, 0.0, 0.0], relative_vel=[0.0, 0.0, 0.0],
          rotation_angles=[0.0, 0.0, 0.0], shift_to_com=False,
          write_new_snapshot=True, include_halo_zero=True,
          metallicity_in_everything=False, arepo=False):
    
    """ 
    Joins snapshots and writes the result as a new snapshot if
    writeNewSnapshot is True.

    The rotation will be applied to the second snapshot. Angles should
    be given in degrees.
    
    Rotation is applied using a for loop that transforms each vector
    with the function rotation().
    """


    relative_pos = [float(i) for i in relative_pos]
    relative_vel = [float(i) for i in relative_vel]

    rotation_angles = np.radians([float(i) for i in rotation_angles])

    #standard families in gadget 2
    #particle_type_len = 6
    #particle_types = [f"PartType{i}" for i in range(particle_type_len)]

    #Loading snapshots
    snapshot_zero = h5py.File(snapshot_zero, "r")
    snapshot_one = h5py.File(snapshot_one, "r")
    print("Snapshots loaded!")


    print("Setting output structure...")
    master_structure = get_master_structure(snapshot_zero, snapshot_one)

    # Header group isn't important to join everything

    # Getting information and joining each type of particle

    all_data = {i:{} for i in master_structure}
    for current_group in master_structure:

        print(f"Joining {current_group}...")

        if current_group != "PartType1" or include_halo_zero: # skips writing the halo from the first snapshot if includeHaloZero is False
            exists_in_zero = current_group in snapshot_zero
        else:
            exists_in_zero = False
                
        exists_in_one = current_group in snapshot_one

        

        if exists_in_zero and exists_in_one:
            # TODO account for missing datasets in any of the snapshots

            all_data[current_group] = solve_group(snapshot_zero[current_group],
                                                  snapshot_one[current_group],
                                                  master_structure[current_group],
                                                  rotation_angles,
                                                  relative_pos,
                                                  relative_vel) # all keys in output


        elif exists_in_zero:
            for current_dataset in snapshot_zero[current_group]:
                all_data[current_group][current_dataset] = snapshot_zero[current_group][current_dataset][:]
                
        elif exists_in_one:
            for current_dataset in snapshot_one[current_group]:
                all_data[current_group][current_dataset] = snapshot_one[current_group][current_dataset][:]
            
            # exception for position and velocity, which should be rotated and translated
            if "Coordinates" in snapshot_one[current_group] and "Velocities" in snapshot_one[current_group]:
                current_coordinates = all_data[current_group]["Coordinates"]
                current_velocities = all_data[current_group]["Velocities"]
                
                if np.any(rotation_angles):
                    current_coordinates = np.array([rotation(h, *rotation_angles) for h in current_coordinates])
                    current_velocities = np.array([rotation(h, *rotation_angles) for h in current_velocities])
                
                all_data[current_group]["Coordinates"] = current_coordinates + relative_pos
                all_data[current_group]["Velocities"] = current_velocities + relative_vel

        else:
            pass
            # should this part exist?


    if shift_to_com:
        total_mass = 0
        pondered_coordinates = []
        for current_group in master_structure:
            current_coordinates = all_data[current_group]["Coordinates"]
            current_masses = all_data[current_group]["Masses"]

            pondered_coordinates.append(
                np.sum(current_masses[:, np.newaxis] * current_coordinates, axis=0))
            total_mass += np.sum(current_masses)
            #print(total_mass)

        center_of_mass = np.sum(np.array(pondered_coordinates) / total_mass, axis=0)
        
        print("The center of mass is located at",
              f"{center_of_mass[0]:0.2f} {center_of_mass[1]:0.2f} {center_of_mass[2]:0.2f}. ",
              "Shifting to it...")
        
        for current_group in master_structure:
            all_data[current_group]["Coordinates"] -= center_of_mass


    all_data, npart = fix_ids_and_npart(all_data)

    #nPart needs to be a list with {partycle_type_len} objects
    #nPart needs to be a list with 6 objects when using default options

    # TODO identify which particle types are here

    indexes = []
    for i in master_structure:
        indexes.append(int(i[-1]))

    final_npart = np.zeros(6, dtype="uint32") # TODO fix hardcode for npart size

    for i, j in zip(indexes, npart):
        final_npart[i] = j
        #print(i, j)


    print(f"Writing snapshot as {output}...")
    write_snapshot(all_data, output, final_npart)
    print("Done!")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Self contained program\
     written in python 3 that can join snapshots using hdf5 format. Based in\
     the module \"snapwrite\" contained in Rafael Ruggiero\"s IC generation\
     programs, but modified to work in python 3.")

    parser.add_argument("snapshot0", help="The name of the first input file.")
    parser.add_argument("snapshot1", help="The name of the second input file.")
    parser.add_argument("-o", "--output", default="init.hdf5", help="The name of\
                        the output file")
    parser.add_argument("-rP", "--relative-position", nargs=3,
                        metavar=("X", "Y", "Z"), default=[0.0, 0.0, 0.0],
                        help="Relative position of the second snapshot with\
                         relation to the first. Must be given in terms of it\"s\
                         components in x, y and z")
    parser.add_argument("-rV", "--relative-velocity", nargs=3,
                        metavar=("vX", "vY", "vZ"), default=[0.0, 0.0, 0.0],
                        help="Relative velocity of the second snapshot with\
                         relation to the first. Must be given in terms of it\"s\
                         components in x, y and z")
    parser.add_argument("-r", "--rotation", nargs=3,
                        metavar=("angleX", "angleY", "angleZ"),
                        default=[0.0, 0.0, 0.0], help="Angles (in degrees) of\
                         the rotation to be applied to the second snapshot.\
                         Must be given in terms of rotations around the x\",\
                         y\" and z\" axis that pass by the origin of the second\
                         snapshot")
    parser.add_argument("--noMainHalo", action="store_false", help="This will\
                         make the program skip the halo of dark matter in the\
                         first snapshot given.")
    parser.add_argument("--COMshift", action="store_true", help="This will\
                         shift the resultng snapshot into into the center of\
                         mass.")

    args = parser.parse_args()


    join(args.snapshot0, args.snapshot1, args.output,
         relative_pos=args.relative_position,
         relative_vel=args.relative_velocity,
         rotation_angles=args.rotation,
         include_halo_zero=args.noMainHalo,
         shift_to_com=args.COMshift)




