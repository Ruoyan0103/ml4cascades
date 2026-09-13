import numpy as np
from ase import units
from ase.io import write
from ase.io.lammpsrun import read_lammps_dump_text


def lammps_dump_to_extxyz(dump_file, out_file, specorder, index=-1, fixed_atom_ids=None):
    """Convert a LAMMPS custom dump (metal units: Angstrom, ps, eV) into an
    extxyz file with the species/pos/velocities/fix_atoms columns TurboGAP's
    ``atoms_file`` expects.

    index selects which dump frame to convert (default: the last one, e.g.
    for building a restart atoms_file from the final frame).
    """
    with open(dump_file) as f:
        atoms = read_lammps_dump_text(f, index=index, specorder=specorder)

    # read_lammps_dump_text already interprets the dump as metal units and
    # stores the result as momenta (velocity * mass) in ASE's native velocity
    # unit (Angstrom per ASE time unit, ~10.18 fs). get_velocities() = momenta
    # / masses gives that native-unit velocity back; multiplying by
    # ase.units.fs converts it to Angstrom/fs, which is what TurboGAP expects.
    velocities = atoms.get_velocities() * units.fs

    n_atoms = len(atoms)
    fix_flags = np.full((n_atoms, 3), 'F', dtype='<U1')
    if fixed_atom_ids:
        fix_flags[list(fixed_atom_ids)] = 'T'

    atoms.set_array('velocities', velocities)
    atoms.new_array('fix_atoms', fix_flags)
    del atoms.arrays['momenta']  # leftover from read_lammps_dump_text; would otherwise
                                 # leak into any later extxyz write that doesn't pass
                                 # an explicit columns= list (e.g. run_cascade's copy/write)

    write(
        out_file,
        atoms,
        format='extxyz',
        columns=['symbols', 'positions', 'velocities', 'fix_atoms'],
    )

    return atoms
