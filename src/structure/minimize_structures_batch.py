#!/usr/bin/env python3
"""
Minimize PDB structures using the same OpenMM energy-minimization pipeline as
ABodyBuilder2 / ImmuneBuilder (amber14/protein.ff14SB.xml, in-vacuo).

Designed for post-processing ABB3 outputs, but works on any heavy-chain /
light-chain PDB.  Run under the *abb2* conda environment (which has openmm,
pdbfixer, scipy).

Usage:
    python3 minimize_structures_batch.py \\
        --input-dir  /path/to/dataset_abb3_1 \\
        --output-dir /path/to/dataset_abb3_1_minimized \\
        [--n-threads 1] [--jobs 8] [--skip-existing]

Called by predict_structure.sh; not normally invoked directly.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pdbfixer
from openmm import (
    LangevinIntegrator,
    CustomExternalForce,
    CustomTorsionForce,
    OpenMMException,
    Platform,
    app,
    unit,
)
from scipy import spatial

# ---------------------------------------------------------------------------
# Constants (identical to ImmuneBuilder/refine.py)
# ---------------------------------------------------------------------------

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms
spring_unit = ENERGY / (LENGTH**2)

CLASH_CUTOFF = 0.63
atom_radii = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
radii_sums = {
    i + j: atom_radii[i] + atom_radii[j]
    for i in atom_radii
    for j in atom_radii
}
cutoffs = {k: CLASH_CUTOFF * v for k, v in radii_sums.items()}

forcefield = app.ForceField("amber14/protein.ff14SB.xml")


# ---------------------------------------------------------------------------
# Refine logic — copied verbatim from ImmuneBuilder/refine.py
# ---------------------------------------------------------------------------

def minimize_energy(topology, positions, k1=2.5, k2=2.5, n_threads=-1):
    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(modeller.topology)

    force = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", k1 * spring_unit)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)
    for residue in modeller.topology.residues():
        for atom in residue.atoms():
            if atom.name in ["CA", "CB", "N", "C"]:
                force.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(force)

    if k2 > 0.0:
        cis_force = CustomTorsionForce("10*k2*(1+cos(theta))^2")
        cis_force.addGlobalParameter("k2", k2 * ENERGY)
        for chain in modeller.topology.chains():
            residues = list(chain.residues())
            rel = [
                {a.name: a.index for a in res.atoms() if a.name in ["N", "CA", "C"]}
                for res in residues
            ]
            for i in range(1, len(residues)):
                if residues[i].name == "PRO":
                    continue
                cis_force.addTorsion(
                    rel[i - 1]["CA"], rel[i - 1]["C"], rel[i]["N"], rel[i]["CA"]
                )
        system.addForce(cis_force)

    integrator = LangevinIntegrator(0, 0.01, 0.0)
    if n_threads > 0:
        platform = Platform.getPlatformByName("CPU")
        simulation = app.Simulation(
            modeller.topology, system, integrator,
            platform, {"Threads": str(n_threads)},
        )
    else:
        simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    return simulation


def chirality_fixer(simulation):
    topology = simulation.topology
    positions = simulation.context.getState(getPositions=True).getPositions()

    d_stereoisomers = []
    for residue in topology.residues():
        if residue.name == "GLY":
            continue
        atom_indices = {
            a.name: a.index
            for a in residue.atoms()
            if a.name in ["N", "CA", "C", "CB"]
        }
        vectors = [
            positions[atom_indices[i]] - positions[atom_indices["CA"]]
            for i in ["N", "C", "CB"]
        ]
        if np.dot(np.cross(vectors[0], vectors[1]), vectors[2]) < 0.0 * LENGTH**3:
            indices = {x.name: x.index for x in residue.atoms() if x.name in ["HA", "CA"]}
            positions[indices["HA"]] = (
                2 * positions[indices["CA"]] - positions[indices["HA"]]
            )
            particle_mass = simulation.system.getParticleMass(indices["HA"])
            simulation.system.setParticleMass(indices["HA"], 0.0)
            d_stereoisomers.append((indices["HA"], particle_mass))

    if d_stereoisomers:
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()
        for atom in d_stereoisomers:
            simulation.system.setParticleMass(*atom)
        simulation.minimizeEnergy()
    return simulation


def bond_check(topology, positions):
    for chain in topology.chains():
        residues = [
            {a.name: a.index for a in res.atoms() if a.name in ["N", "C"]}
            for res in chain.residues()
        ]
        for i in range(len(residues) - 1):
            v = np.linalg.norm(
                positions[residues[i]["C"]] - positions[residues[i + 1]["N"]]
            )
            if abs(v - 1.329 * LENGTH) > 0.1 * LENGTH:
                return False
    return True


def cis_bond(p0, p1, p2, p3):
    ab = p1 - p0
    cd = p2 - p1
    db = p3 - p2
    u = np.cross(-ab, cd)
    v = np.cross(db, cd)
    return np.dot(u, v) > 0


def cis_check(topology, positions):
    pos = np.array(positions.value_in_unit(LENGTH))
    for chain in topology.chains():
        residues = list(chain.residues())
        rel = [
            {a.name: a.index for a in res.atoms() if a.name in ["N", "CA", "C"]}
            for res in residues
        ]
        for i in range(1, len(residues)):
            if residues[i].name == "PRO":
                continue
            r, nr = rel[i - 1], rel[i]
            if cis_bond(pos[r["CA"]], pos[r["C"]], pos[nr["N"]], pos[nr["CA"]]):
                return False
    return True


def stereo_check(topology, positions):
    pos = np.array(positions.value_in_unit(LENGTH))
    for residue in topology.residues():
        if residue.name == "GLY":
            continue
        idx = {
            a.name: a.index
            for a in residue.atoms()
            if a.name in ["N", "CA", "C", "CB"]
        }
        vecs = pos[[idx[i] for i in ["N", "C", "CB"]]] - pos[idx["CA"]]
        if np.linalg.det(vecs) < 0:
            return False
    return True


def clash_check(topology, positions):
    heavies = [x for x in topology.atoms() if x.element.symbol != "H"]
    pos = np.array(positions.value_in_unit(LENGTH))[[x.index for x in heavies]]
    tree = spatial.KDTree(pos)
    pairs = tree.query_pairs(r=max(cutoffs.values()))
    for pair in pairs:
        ai, aj = heavies[pair[0]], heavies[pair[1]]
        if ai.residue.index == aj.residue.index:
            continue
        if (ai.name == "C" and aj.name == "N") or (ai.name == "N" and aj.name == "C"):
            continue
        d = np.linalg.norm(pos[pair[0]] - pos[pair[1]])
        if ai.name == "SG" and aj.name == "SG" and d > 1.88:
            continue
        if d < cutoffs[ai.element.symbol + aj.element.symbol]:
            return False
    return True


def strained_sidechain_bonds_check(topology, positions):
    atoms = list(topology.atoms())
    pos = np.array(positions.value_in_unit(LENGTH))
    system = forcefield.createSystem(topology)
    bonds = [x for x in system.getForces() if type(x).__name__ == "HarmonicBondForce"][0]
    n_bonds = bonds.getNumBonds()
    ii = np.empty(n_bonds, dtype=int)
    jj = np.empty(n_bonds, dtype=int)
    k = np.empty(n_bonds)
    x0 = np.empty(n_bonds)
    for n in range(n_bonds):
        ii[n], jj[n], _x0, _k = bonds.getBondParameters(n)
        k[n] = _k.value_in_unit(spring_unit)
        x0[n] = _x0.value_in_unit(LENGTH)
    distance = np.linalg.norm(pos[ii] - pos[jj], axis=-1)
    check = k * (distance - x0) ** 2 > 100
    return [atoms[x].residue for x in ii[check]]


def strained_sidechain_bonds_fixer(strained_residues, topology, positions, n_threads=-1):
    bb_atoms = ["N", "CA", "C"]
    bad_side_chains = [
        atom
        for residue in strained_residues
        for atom in residue.atoms()
        if atom.name not in bb_atoms
    ]
    modeller = app.Modeller(topology, positions)
    modeller.delete(bad_side_chains)

    random_number = str(int(np.random.rand() * 10**8))
    tmp_file = f"side_chain_fix_tmp_{random_number}.pdb"
    with open(tmp_file, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

    fixer = pdbfixer.PDBFixer(tmp_file)
    os.remove(tmp_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield)
    system = forcefield.createSystem(modeller.topology)
    integrator = LangevinIntegrator(0, 0.01, 0.0)
    if n_threads > 0:
        platform = Platform.getPlatformByName("CPU")
        simulation = app.Simulation(
            modeller.topology, system, integrator,
            platform, {"Threads": str(n_threads)},
        )
    else:
        simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    return simulation.topology, simulation.context.getState(getPositions=True).getPositions()


def refine_once(input_file, output_file, check_for_strained_bonds=True, n=6, n_threads=-1):
    k1s = [2.5, 1, 0.5, 0.25, 0.1, 0.001]
    k2s = [2.5, 5, 7.5, 15, 25, 50]
    success = False

    fixer = pdbfixer.PDBFixer(input_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    k1 = k1s[0]
    k2 = -1 if cis_check(fixer.topology, fixer.positions) else k2s[0]
    topology, positions = fixer.topology, fixer.positions

    for i in range(n):
        try:
            simulation = minimize_energy(topology, positions, k1=k1, k2=k2, n_threads=n_threads)
            topology, positions = (
                simulation.topology,
                simulation.context.getState(getPositions=True).getPositions(),
            )
            acceptable_bonds = bond_check(topology, positions)
            trans_peptide_bonds = cis_check(topology, positions)
        except OpenMMException as e:
            if i == n - 1 and "positions" not in locals():
                print(f"OpenMM failed to refine {input_file}", flush=True)
                raise e
            topology, positions = fixer.topology, fixer.positions
            continue

        if not acceptable_bonds:
            k1 = k1s[min(i, len(k1s) - 1)]
        if not trans_peptide_bonds:
            k2 = k2s[min(i, len(k2s) - 1)]
        else:
            k2 = -1

        if acceptable_bonds and trans_peptide_bonds:
            try:
                simulation = chirality_fixer(simulation)
                topology, positions = (
                    simulation.topology,
                    simulation.context.getState(getPositions=True).getPositions(),
                )
            except OpenMMException:
                topology, positions = fixer.topology, fixer.positions
                continue

            if check_for_strained_bonds:
                try:
                    strained_bonds = strained_sidechain_bonds_check(topology, positions)
                    if len(strained_bonds) > 0:
                        needs_recheck = True
                        topology, positions = strained_sidechain_bonds_fixer(
                            strained_bonds, topology, positions, n_threads=n_threads
                        )
                    else:
                        needs_recheck = False
                except OpenMMException:
                    topology, positions = fixer.topology, fixer.positions
                    continue
            else:
                needs_recheck = False

            tests = bond_check(topology, positions) and cis_check(topology, positions)
            if needs_recheck:
                tests = tests and strained_sidechain_bonds_check(topology, positions)
            if tests and stereo_check(topology, positions) and clash_check(topology, positions):
                success = True
                break

    with open(output_file, "w") as out_handle:
        app.PDBFile.writeFile(topology, positions, out_handle, keepIds=True)
    return success


def refine(input_file, output_file, check_for_strained_bonds=True, tries=3, n=6, n_threads=-1):
    for _ in range(tries):
        if refine_once(
            input_file, output_file,
            check_for_strained_bonds=check_for_strained_bonds,
            n=n, n_threads=n_threads,
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-PDB worker (called in a worker process)
# ---------------------------------------------------------------------------

def _minimize_one(task: tuple) -> tuple[str, str, bool]:
    """Returns (input_path, status_str, success_bool)."""
    input_pdb, output_pdb, n_threads, skip_existing = task
    if skip_existing and os.path.exists(output_pdb):
        return input_pdb, "skipped", True
    try:
        success = refine(input_pdb, output_pdb, n_threads=n_threads)
        status = "ok" if success else "warn"
        return input_pdb, status, success
    except Exception as e:
        return input_pdb, f"FAIL: {e}", False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, metavar="DIR",
        help="Folder of PDB files to minimize.",
    )
    parser.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Output folder (created if needed).",
    )
    parser.add_argument(
        "--n-threads", type=int, default=1, metavar="N",
        help="OpenMM CPU threads per job (default: 1).",
    )
    parser.add_argument(
        "--jobs", type=int, default=8, metavar="J",
        help="Number of PDB files to minimize in parallel (default: 8).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip PDBs already present in --output-dir.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        sys.exit(f"Not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"No PDB files found in {input_dir}", flush=True)
        return

    print(
        f"Minimizing {len(pdb_files)} structure(s): {input_dir} -> {output_dir}",
        flush=True,
    )

    tasks = [
        (str(p), str(output_dir / p.name), args.n_threads, args.skip_existing)
        for p in pdb_files
    ]

    n_ok = n_warn = n_skip = n_fail = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_minimize_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            name, status, _ = future.result()
            short = Path(name).name
            if status == "skipped":
                n_skip += 1
            elif status == "ok":
                n_ok += 1
                print(f"  ok   {short}", flush=True)
            elif status == "warn":
                n_warn += 1
                print(f"  warn {short}  (did not fully converge)", flush=True)
            else:
                n_fail += 1
                print(f"  {status}  {short}", flush=True)

    print(
        f"\nDone: {n_ok} ok, {n_warn} warn, {n_skip} skipped, {n_fail} failed.",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
