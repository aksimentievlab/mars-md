"""The ``.bd`` keyword tables and the ownership map.

Every key the C++ parser recognizes appears here exactly once, mapped to the
model field it fills and the API the applier calls. Keys the engine knows but
ignores are listed too, so they land in ``BdConfig.unsupported`` rather than
tripping strict mode.

Source of truth: ``src/IO/ConfigParser.cpp`` -- ``parse_parameters`` for the
globals, ``get_elements`` for blocks and top-level element keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "Owner",
    "GLOBAL_KEYS",
    "PARTICLE_FIELD_KEYS",
    "RIGID_BODY_FIELD_KEYS",
    "RIGID_BODY_PMF_KEYS",
    "TOP_LEVEL_ELEMENT_KEYS",
    "KNOWN_UNSUPPORTED_KEYS",
    "BLOCK_HEADERS",
    "TOPOLOGY_KEYS",
    "ANALYTICAL_BOND_TYPES",
    "ANALYTICAL_ANGLE_TYPES",
    "ANALYTICAL_DIHEDRAL_TYPES",
    "PERIODICITY_MAP",
    "DECOMPOSER_MAP",
    "LONG_RANGE_MAP",
    "INTEGRATOR_MAP",
    "OUTPUT_FORMAT_MAP",
    "canonical_key",
    "is_known_key",
]


@dataclass(frozen=True, slots=True)
class Owner:
    """One row of the ownership map.

    :param field: attribute on the model dataclass that holds the parsed value.
    :param applies_to: the ``_core`` API the applier ultimately calls, or
        ``""`` when the key is parsed but deliberately not applied.
    """

    field: str
    applies_to: str = ""


BLOCK_HEADERS: Final[frozenset[str]] = frozenset({"particle", "rigidBody"})


# ---------------------------------------------------------------------------
# globals -- ConfigParser::parse_parameters
# ---------------------------------------------------------------------------
# Each entry maps every accepted spelling to one canonical key. The C++ checks
# camelCase then snake_case via findParameterVariant(); several calls pass the
# same string twice, so those keys have exactly one spelling.

GLOBAL_KEYS: Final[dict[str, Owner]] = {
    "temperature": Owner("temperature", "SimSystem.set_temperature_value"),
    # ConfigParser.cpp:149 passes the snake name as *both* variants, so the
    # camelCase spelling is unreachable in the engine. Accepted here anyway.
    "temperature_grid": Owner("temperature_grid", "SimSystem.set_temperature_grid"),
    "temperatureGrid": Owner("temperature_grid", "SimSystem.set_temperature_grid"),
    "seed": Owner("seed", "SimSystem.set_base_seed"),
    "cutoff": Owner("cutoff", "SimSystem.set_cutoff"),
    "pairlistDistance": Owner("pairlist_distance", "SimSystem.set_pairlist_cutoff"),
    "pairlist_distance": Owner("pairlist_distance", "SimSystem.set_pairlist_cutoff"),
    "timestep": Owner("timestep", "SimSystem.set_timestep"),
    "steps": Owner("steps", "SimSystem.set_num_steps"),
    "decompPeriod": Owner(
        "pairlist_rebuild_period", "SimSystem.set_neighbor_list_rebuild_period"
    ),
    "pairlist_rebuild_period": Owner(
        "pairlist_rebuild_period", "SimSystem.set_neighbor_list_rebuild_period"
    ),
    "reorderPeriod": Owner("reorder_period", "SimSystem.set_reorder_period"),
    "reorder_period": Owner("reorder_period", "SimSystem.set_reorder_period"),
    "outputPeriod": Owner("output_period", "SimSystem.set_output_period"),
    "output_period": Owner("output_period", "SimSystem.set_output_period"),
    "outputEnergyPeriod": Owner(
        "output_energy_period", "SimSystem.set_energy_output_period"
    ),
    "output_energy_period": Owner(
        "output_energy_period", "SimSystem.set_energy_output_period"
    ),
    "outputName": Owner("output_name", "SimSystem.set_output_name"),
    "output_name": Owner("output_name", "SimSystem.set_output_name"),
    "outputFormat": Owner("output_format", "SimSystem.set_output_format"),
    "output_format": Owner("output_format", "SimSystem.set_output_format"),
    "systemSize": Owner("system_size", "SimSystem.set_box_size"),
    "system_size": Owner("system_size", "SimSystem.set_box_size"),
    "origin": Owner("origin", "SimSystem.set_origin"),
    "decomposer": Owner("decomposer", "SimSystem.set_decomposer_type"),
    "longRangeMethod": Owner("long_range_method", "SimSystem.set_long_range_method"),
    "long_range_method": Owner("long_range_method", "SimSystem.set_long_range_method"),
    # Three spellings, one setter (ConfigParser.cpp:254-272).
    "ParticleDynamicType": Owner(
        "particle_dynamic_type", "SimSystem.set_particle_integrator_type"
    ),
    "particleDynamicType": Owner(
        "particle_dynamic_type", "SimSystem.set_particle_integrator_type"
    ),
    "particle_dynamic_type": Owner(
        "particle_dynamic_type", "SimSystem.set_particle_integrator_type"
    ),
    "algorithm": Owner(
        "particle_dynamic_type", "SimSystem.set_particle_integrator_type"
    ),
    "RigidBodyDynamicType": Owner(
        "rigid_body_dynamic_type", "SimSystem.set_rigid_body_integrator_type"
    ),
    "rigidBodyDynamicType": Owner(
        "rigid_body_dynamic_type", "SimSystem.set_rigid_body_integrator_type"
    ),
    "rigid_body_dynamic_type": Owner(
        "rigid_body_dynamic_type", "SimSystem.set_rigid_body_integrator_type"
    ),
    "rigidBodyAlgorithm": Owner(
        "rigid_body_dynamic_type", "SimSystem.set_rigid_body_integrator_type"
    ),
    "rigid_body_algorithm": Owner(
        "rigid_body_dynamic_type", "SimSystem.set_rigid_body_integrator_type"
    ),
}


# ---------------------------------------------------------------------------
# particle block -- ConfigParser.cpp:592-606
# ---------------------------------------------------------------------------

PARTICLE_FIELD_KEYS: Final[dict[str, Owner]] = {
    "num": Owner("num", "ParticleType.num"),
    "diffusion": Owner("diffusion", "ParticleType.diffusivity"),
    "transDamping": Owner("trans_damping", "ParticleType.damping_coefficient"),
    "mass": Owner("mass", "ParticleType.mass"),
    "gridFile": Owner("grid_files", "GridManager.add_dense_grid + pmf_grids"),
    "diffusionGridFile": Owner(
        "diffusion_grid_file", "ParticleType.diffusion_grid_name"
    ),
    "forceGridFiles": Owner("force_grid_files", "ParticleType.force_grid_names"),
    "forceGridScale": Owner("force_grid_scale", "ParticleType.force_grid_scale"),
    # Deferred lists: applied after the block closes, so they may appear before
    # or after the gridFile they scale.
    "gridFileScale": Owner("grid_file_scale", "GridTerm.scale"),
    "gridFileScaleSlope": Owner("grid_file_scale_slope", "GridTerm.scale_slope"),
    "gridFileBoundaryConditions": Owner(
        "grid_file_boundary_conditions", "GridTerm.boundary_condition"
    ),
    "gridFileSMD": Owner("grid_file_smd", "ParticleType.pmf_smd_freq"),
    "rigidBodyPotential": Owner(
        "rigid_body_potential_keys", "ParticleType.rigid_body_potential_keys"
    ),
}


# ---------------------------------------------------------------------------
# rigidBody block -- ConfigParser.cpp:608-635
# ---------------------------------------------------------------------------

RIGID_BODY_FIELD_KEYS: Final[dict[str, Owner]] = {
    "mass": Owner("mass", "RigidBodyType.mass"),
    "num": Owner("num", "instance count"),
    "inertia": Owner("inertia", "RigidBodyType.moment_of_inertia"),
    "transDamping": Owner("trans_damping", "RigidBodyType.damping_coefficient"),
    "rotDamping": Owner("rot_damping", "RigidBodyType.rotational_damping"),
    "inputPsf": Owner("input_psf", "load_rigid_body_pdb_psf"),
    "inputPdb": Owner("input_pdb", "load_rigid_body_pdb_psf"),
    "referencePoint": Owner("reference_point", "load_rigid_body_pdb_psf"),
    "densityGrid": Owner("density_grids", "RigidBodyType.density_grids"),
    "potentialGrid": Owner("potential_grids", "RigidBodyType.potential_grids"),
    "pmf": Owner("pmf_grids", "RigidBodyType.pmf_grids"),
    # Migration-only aliases, accepted here and nowhere else: the engine takes
    # `pmf` alone, deliberately, so the 2.0 grammar stays unambiguous. v1 spelled
    # this list `gridFile` inside a rigidBody block (RigidBodyType::addPMF), with
    # the same `<key> <file>` grammar.
    "gridFile": Owner("pmf_grids", "RigidBodyType.pmf_grids"),
    "pmfFile": Owner("pmf_grids", "RigidBodyType.pmf_grids"),
    "densityGridScale": Owner("density_grid_scales", "GridTerm.scale"),
    "potentialGridScale": Owner("potential_grid_scales", "GridTerm.scale"),
    "pmfScale": Owner("pmf_grid_scales", "GridTerm.scale"),
    # Per-instance, not per-type.
    "position": Owner("position", "RigidBody.position"),
    "orientation": Owner("orientation", "RigidBody.orientation"),
    "momentum": Owner("momentum", "RigidBody.momentum"),
    "angularMomentum": Owner("angular_momentum", "RigidBody.angular_momentum"),
    "constantForce": Owner("external_force", "RigidBody.external_force"),
    "constantTorque": Owner("external_torque", "RigidBody.external_torque"),
}

#: Keys that all append to a rigid body's PMF grid list. ``pmf`` is the engine's
#: only spelling; the other two are accepted for v1 input. See
#: :data:`RIGID_BODY_FIELD_KEYS`.
RIGID_BODY_PMF_KEYS: Final[frozenset[str]] = frozenset({"pmf", "pmfFile", "gridFile"})


# ---------------------------------------------------------------------------
# top-level element keys -- ConfigParser.cpp:1029-1084
# ---------------------------------------------------------------------------

#: ``inputBonds`` and friends all route through one reader, since
#: ``BondConfigReader::read_file`` dispatches on the record type inside the
#: file, not on which key named it.
TOPOLOGY_KEYS: Final[dict[str, Owner]] = {
    "inputBonds": Owner("topology_files", "BondedInteractions.add_bond"),
    "input_bonds": Owner("topology_files", "BondedInteractions.add_bond"),
    "inputAngles": Owner("topology_files", "BondedInteractions.add_angle"),
    "input_angles": Owner("topology_files", "BondedInteractions.add_angle"),
    "inputDihedrals": Owner("topology_files", "BondedInteractions.add_dihedral"),
    "input_dihedrals": Owner("topology_files", "BondedInteractions.add_dihedral"),
    "inputExcludes": Owner("topology_files", "BondedInteractions.add_exclusion"),
    "input_excludes": Owner("topology_files", "BondedInteractions.add_exclusion"),
    "inputRestraints": Owner("topology_files", "BondedInteractions.add_restraint"),
    "input_restraints": Owner("topology_files", "BondedInteractions.add_restraint"),
    "inputProductPotentials": Owner("topology_files", ""),
    "input_product_potentials": Owner("topology_files", ""),
}

TOP_LEVEL_ELEMENT_KEYS: Final[dict[str, Owner]] = {
    **TOPOLOGY_KEYS,
    "tabulatedFile": Owner("tabulated_pairs", "TablesRegistry.load_pair_nonbonded"),
    "tabulated_file": Owner("tabulated_pairs", "TablesRegistry.load_pair_nonbonded"),
    "inputParticles": Owner("input_particles", "SimManager.stage_particles"),
    "input_particles": Owner("input_particles", "SimManager.stage_particles"),
    "restartCoordinates": Owner("restart_coordinates", "SimManager.stage_particles"),
    "restart_coordinates": Owner("restart_coordinates", "SimManager.stage_particles"),
    "inputRBCoordinates": Owner(
        "input_rb_coordinates", "SimManager.stage_rigid_bodies"
    ),
    "input_rb_coordinates": Owner(
        "input_rb_coordinates", "SimManager.stage_rigid_bodies"
    ),
    "rigidBodyGridGridPeriod": Owner(
        "rb_grid_grid_period", "SimSystem.set_rb_update_period"
    ),
    "rigid_body_grid_grid_period": Owner(
        "rb_grid_grid_period", "SimSystem.set_rb_update_period"
    ),
}


# ---------------------------------------------------------------------------
# known but not applied
# ---------------------------------------------------------------------------

#: Keys the engine parses past without acting on. They appear throughout the
#: existing fixtures, so treating them as unknown would make strict mode
#: useless. Each lands in ``BdConfig.unsupported``.
KNOWN_UNSUPPORTED_KEYS: Final[dict[str, str]] = {
    "numberFluct": "deprecated; grand-canonical particle count fluctuation",
    "interparticleForce": "deprecated; only the value 1 was ever supported",
    "fullLongRange": "deprecated; superseded by longRangeMethod",
    "tabulatedPotential": "flag only; tabulatedFile entries carry the real data",
    "tabulated_potential": "flag only; tabulatedFile entries carry the real data",
    "electricField": "not wired in the 2.0 parser; no SimSystem setter exists",
    "electric_field": "not wired in the 2.0 parser; no SimSystem setter exists",
    # The topology files name their potential by path on each record line, so
    # these top-level declarations are redundant and the engine ignores them.
    "tabulatedBondFile": "redundant; inputBonds records name the .dat directly",
    "tabulatedAngleFile": "redundant; inputAngles records name the .dat directly",
    "tabulatedDihedralFile": "redundant; inputDihedrals records name the .dat directly",
    # Not in rigid_body_field_keys, so the engine sees it at top level and drops
    # it. Present in Tests/privite_test/trombone-ssb/run.bd.
    "attachedParticles": "not wired; use inputPdb/inputPsf + referencePoint",
    "attached_particles": "not wired; use inputPdb/inputPsf + referencePoint",
    "rigidBodyPotentialFile": "not wired in the 2.0 parser",
    "scaleDensityGrid": "legacy spelling; use densityGridScale",
    "scalePotentialGrid": "legacy spelling; use potentialGridScale",
    "scalePMF": "legacy spelling; use pmfScale",
}


# ---------------------------------------------------------------------------
# enum value tables -- ConfigParser.cpp:112-139, matched case-insensitively
# ---------------------------------------------------------------------------

PERIODICITY_MAP: Final[dict[str, str]] = {
    "allperiodic": "AllPeriodic",
    "twodimensional": "TwoDimensional",
    "onedimensional": "OneDimensional",
    "open": "Open",
}

DECOMPOSER_MAP: Final[dict[str, str]] = {
    "spatial": "Spatial",
    "recursivebisection": "RecursiveBisection",
    "geometric": "Geometric",
}

LONG_RANGE_MAP: Final[dict[str, str]] = {
    "cutoffamr": "CutoffAMR",
    "pppm": "PPPM",
    "pme": "PME",
    "fmm": "FMM",
    "direct": "Direct",
    "none": "None",
}

INTEGRATOR_MAP: Final[dict[str, str]] = {
    "brownian": "Brownian",
    "langevin": "Langevin",
    "velocityverlet": "VelocityVerlet",
}

OUTPUT_FORMAT_MAP: Final[dict[str, str]] = {
    "dcd": "DCD",
    "pdb": "PDB",
    "hdf5": "HDF5",
}


# ---------------------------------------------------------------------------
# analytical potential names -- Interactions/BondedInteraction.h:17-29
# ---------------------------------------------------------------------------
# A topology record whose function name is in one of these lists is Analytical
# and its function_index is the list position. Anything else is Tabulated and
# the name is a file path.

ANALYTICAL_BOND_TYPES: Final[tuple[str, ...]] = (
    "Harmonic",
    "Morse",
    "FENE",
    "Half_Harmonic",
    "WLCSK",
)
ANALYTICAL_ANGLE_TYPES: Final[tuple[str, ...]] = ANALYTICAL_BOND_TYPES
ANALYTICAL_DIHEDRAL_TYPES: Final[tuple[str, ...]] = ANALYTICAL_BOND_TYPES


def canonical_key(key: str) -> str:
    """Return the canonical model field a key fills, or ``""`` if unknown."""
    for table in (
        GLOBAL_KEYS,
        TOP_LEVEL_ELEMENT_KEYS,
        PARTICLE_FIELD_KEYS,
        RIGID_BODY_FIELD_KEYS,
    ):
        owner = table.get(key)
        if owner is not None:
            return owner.field
    return ""


def is_known_key(key: str) -> bool:
    """True if the engine recognizes ``key`` at all, applied or not."""
    return (
        key in BLOCK_HEADERS
        or key in GLOBAL_KEYS
        or key in TOP_LEVEL_ELEMENT_KEYS
        or key in PARTICLE_FIELD_KEYS
        or key in RIGID_BODY_FIELD_KEYS
        or key in KNOWN_UNSUPPORTED_KEYS
    )
