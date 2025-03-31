# snapshotJoinerMk2

Self contained script that joins simulation snapshots in hdf5.

This version only works with HDF5 snapshots, but should be general enough that almost any snapshot combination should work. It differs mainly in this aspect compared to the original snapshotJoiner, which could also join Gadget2 binary snapshots. Since it breaks some compatibility and functionality, a new repository was made.

Has some functions based on "snapwrite" contained in [Rafael Ruggiero](https://ruggiero.github.io/)'s IC generation programs.


## Rotation angles, relative position and velocity

This program applies a rotation to the second snapshot considering angles around its original x, y and z axis. After this rotation, the given relative position and velocity is summed to all position and velocity vectors of the second snapshot.





## Usage Examples

```
python3 snapshotJoiner.py cluster.hdf5 galaxy.hdf5 -o rampressure.hdf5
--relative-position 1000 100 0 --relative-velocity -100 0 0  --rotation 0 90 0
```

```
python3 snapshotJoiner.py clusterA.hdf5 clusterB.hdf5 -o rampressure.hdf5
--relative-position 1000 100 0
```

```
python3 snapshotJoiner.py galaxyA.hdf5 galaxyB.hdf5 -o rampressure.hdf5
--relative-position 1000 0 0 --rotation 0 90 0
```


## Help
This code supports the `--help` flag, that explains each available option.

```
usage: snapshotJoinerMk2.py [-h] [-o OUTPUT] [-rP X Y Z] [-rV vX vY vZ] [-r angleX angleY angleZ] [--noMainHalo] [--COMshift] snapshot0 snapshot1

Self contained program written in python 3 that can join snapshots using hdf5 format. Based in the module "snapwrite" contained in Rafael Ruggiero"s IC generation programs, but modified to work in python 3.

positional arguments:
  snapshot0             The name of the first input file.
  snapshot1             The name of the second input file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        The name of the output file
  -rP X Y Z, --relative-position X Y Z
                        Relative position of the second snapshot with relation to the first. Must be given in terms of it"s components in x, y and z
  -rV vX vY vZ, --relative-velocity vX vY vZ
                        Relative velocity of the second snapshot with relation to the first. Must be given in terms of it"s components in x, y and z
  -r angleX angleY angleZ, --rotation angleX angleY angleZ
                        Angles (in degrees) of the rotation to be applied to the second snapshot. Must be given in terms of rotations around the x", y" and z" axis that pass by the origin of the second snapshot
  --noMainHalo          This will make the program skip the halo of dark matter in the first snapshot given.
  --COMshift            This will shift the resultng snapshot into into the center of mass.
```
