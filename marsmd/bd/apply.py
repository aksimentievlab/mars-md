"""Turn a parsed :class:`~marsmd.bd.model.BdConfig` into engine objects.

This is the only module that constructs ``marsmd._core`` objects, and the only
one that opens the files the parser merely named. Importing it does not require
a built extension -- the import is deferred to first use, so
:func:`resolve_inputs` stays usable on an unbuilt tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from . import aux
from ..staging import fold_in_attached_particles
from .model import BdConfig, ParticleBlock, RigidBodyBlock
from .paths import resolve_file_path

__all__ = ["ResolvedInputs", "resolve_inputs", "BdApplier", "BdSimulation"]


def _core() -> ModuleType:
    """Import the extension on first use, keeping this module import-safe."""
    from .. import _load_core

    core = _load_core()
    assert core is not None  # _load_core raises when it cannot import
    return core


@dataclass(slots=True)
class ResolvedInputs:
    """Auxiliary data loaded from the files a config names.

    :param particles: from ``inputParticles`` or ``restartCoordinates``.
    :param topology: the union of every ``input*`` topology file, in the order
        the config named them.
    :param rigid_body_coords: from ``inputRBCoordinates``; overwrites the
        per-block position/orientation one instance at a time.
    :param grid_paths: every grid filename the config names, mapped to its
        resolved path. Keys stay exactly as written -- they are the
        ``GridManager`` map keys.
    """

    particles: list[aux.ParticleRecord] = field(default_factory=list)
    topology: aux.Topology = field(default_factory=aux.Topology)
    rigid_body_coords: list[aux.RigidBodyCoord] = field(default_factory=list)
    grid_paths: dict[str, str] = field(default_factory=dict)
    tabulated_paths: dict[str, str] = field(default_factory=dict)


def resolve_inputs(config: BdConfig, *, load: bool = True) -> ResolvedInputs:
    """Resolve every filename a config names, and optionally read them.

    :param config: a parsed config.
    :param load: when False, only the path maps are filled -- nothing is
        opened. This is what ``--dry-run`` uses.
    :returns: the auxiliary data the applier needs.

    :raises FileNotFoundError: if ``load`` is set and a named file is missing.
    """
    source = config.source_path
    out = ResolvedInputs()

    def resolve(name: str) -> str:
        return resolve_file_path(name, source)

    for block in config.particles:
        for entry in block.grid_files:
            out.grid_paths[entry.filename] = resolve(entry.filename)
        if block.diffusion_grid_file:
            out.grid_paths[block.diffusion_grid_file] = resolve(
                block.diffusion_grid_file
            )
        for name in block.force_grid_files:
            out.grid_paths[name] = resolve(name)

    for rb in config.rigid_bodies:
        for group in (rb.potential_grids, rb.density_grids, rb.pmf_grids):
            for entry in group:
                out.grid_paths[entry.filename] = resolve(entry.filename)

    for pair in config.tabulated_pairs:
        out.tabulated_paths[pair.filename] = resolve(pair.filename)

    if not load:
        return out

    type_names = [block.name for block in config.particles]

    if config.input_particles is not None:
        out.particles = aux.read_particles_file(resolve(config.input_particles))
    elif config.restart_coordinates is not None:
        out.particles = aux.read_restart_file(
            resolve(config.restart_coordinates), type_names
        )

    for entry in config.topology_files:
        aux.read_topology_file(resolve(entry.filename), out.topology)

    if config.input_rb_coordinates is not None:
        out.rigid_body_coords = aux.read_rigid_body_coords(
            resolve(config.input_rb_coordinates)
        )

    return out


class BdApplier:
    """Apply a parsed config to one ``SimSystem``, and build its staging payloads.

    :func:`apply_to` populates a system in place -- the same job the C++
    ``ConfigParser`` does, and the reason that class takes a ``SimSystem&``
    rather than returning anything. The ``build_*`` methods produce the three
    per-run payloads separately, because those are staged on a ``SimManager``,
    not stored on the system.
    """

    def __init__(self, config: BdConfig, resolved: ResolvedInputs | None = None):
        self.config = config
        self.resolved = resolve_inputs(config) if resolved is None else resolved

    # -- SimSystem population -------------------------------------------------

    def apply_to(self, system) -> None:
        """Populate ``system`` from the config. Mutates in place."""
        self._apply_globals(system)
        for block in self.config.particles:
            system.add_particle_type(self._particle_type(block, system))
        for block in self.config.rigid_bodies:
            system.add_rigid_body_type(self._rigid_body_type(block, system))
        self._apply_tabulated(system)

    def _apply_globals(self, system) -> None:
        core = _core()
        g = self.config.globals

        if g.temperature_grid is not None:
            raise NotImplementedError(
                "temperatureGrid needs a Grid built from a .dx file, and no DX "
                "reader is bound (SimSystem.set_temperature_grid takes a Grid, "
                "and Grid only builds from numpy). Use a scalar temperature."
            )
        system.set_temperature_value(g.temperature)

        system.set_cutoff(g.cutoff)
        # pairlistDistance is a skin added to the cutoff, never an absolute.
        system.set_pairlist_cutoff(g.pairlist_cutoff)
        system.set_timestep(g.timestep)
        system.set_num_steps(g.steps)
        system.set_neighbor_list_rebuild_period(g.pairlist_rebuild_period)
        system.set_output_period(g.output_period)
        system.set_energy_output_period(g.output_energy_period)
        system.set_output_name(g.output_name)

        if g.system_size is not None:
            system.set_box_size(*g.system_size)
        if g.origin is not None:
            system.set_origin(*g.origin)

        # Unset key -> engine default stands. Enums arrive as member names.
        for value, setter, enum in (
            (g.seed, system.set_base_seed, None),
            (g.reorder_period, system.set_reorder_period, None),
            (g.rb_grid_grid_period, system.set_rb_update_period, None),
            (g.output_format, system.set_output_format, core.OutputFormat),
            (g.decomposer, system.set_decomposer_type, core.DecomposerType),
            (g.long_range_method, system.set_long_range_method, core.LongRangeMethod),
            (g.particle_dynamic_type, system.set_particle_integrator_type, core.IntegratorType),
            (
                g.rigid_body_dynamic_type,
                system.set_rigid_body_integrator_type,
                core.IntegratorType,
            ),
        ):
            if value is not None:
                setter(value if enum is None else getattr(enum, value))

    def _grid_terms(self, system, entries) -> tuple[list, list[str]]:
        """Load each entry into the GridManager, returning terms and their keys."""
        core = _core()
        manager = system.get_grid_manager()
        terms, keys = [], []
        for entry in entries:
            key = manager.add_dense_grid(self.resolved.grid_paths[entry.filename])
            if not key.is_valid():
                continue
            terms.append(
                core.GridTerm(
                    key, entry.scale, entry.scale_slope, entry.boundary_condition
                )
            )
            keys.append(entry.key)
        return terms, keys

    def _particle_type(self, block: ParticleBlock, system):
        core = _core()
        pt = core.ParticleType(block.name)
        pt.num = block.num
        pt.mass = block.mass
        pt.pmf_smd_freq = block.grid_file_smd
        pt.rigid_body_potential_keys = list(block.rigid_body_potential_keys)
        if block.diffusion is not None:
            pt.diffusivity = block.diffusion
        if block.trans_damping is not None:
            pt.damping_coefficient = block.trans_damping

        # A type's grid names must be the keys the grids were registered under:
        # init() re-derives every grid id from them (see dev_notes.md).
        manager = system.get_grid_manager()
        paths = self.resolved.grid_paths
        pt.pmf_grids, _ = self._grid_terms(system, block.grid_files)
        pt.pmf_grid_names = [
            paths[e.filename] for e in block.grid_files if manager.has_grid(paths[e.filename])
        ]

        if block.diffusion_grid_file is not None:
            key = manager.add_dense_grid(paths[block.diffusion_grid_file])
            if key.is_valid():
                pt.diffusion_grid_id = key.grid_id
                pt.diffusion_grid_name = paths[block.diffusion_grid_file]

        # force_grid_names is a fixed std::array<3>; the engine takes x/y/z or nothing.
        force = [paths[name] for name in block.force_grid_files]
        if len(force) == 3:
            ids = []
            for path in force:
                key = manager.add_dense_grid(path)
                ids.append(key.grid_id if key.is_valid() else -1)
            pt.force_grid_names = force
            pt.force_grid_id = ids
            if block.force_grid_scale is not None:
                pt.force_grid_scale = block.force_grid_scale
        return pt

    def _rigid_body_type(self, block: RigidBodyBlock, system):
        core = _core()
        rbt = core.RigidBodyType(block.name)
        rbt.mass = block.mass
        if block.inertia is not None:
            rbt.moment_of_inertia = block.inertia
        if block.trans_damping is not None:
            rbt.damping_coefficient = block.trans_damping
        if block.rot_damping is not None:
            rbt.rotational_damping = block.rot_damping

        for entries, terms_attr, keys_attr in (
            (block.density_grids, "density_grids", "density_grid_keys"),
            (block.potential_grids, "potential_grids", "potential_grid_keys"),
            (block.pmf_grids, "pmf_grids", "pmf_keys"),
        ):
            terms, keys = self._grid_terms(system, entries)
            setattr(rbt, terms_attr, terms)
            setattr(rbt, keys_attr, keys)

        if block.input_pdb and block.input_psf:
            core.load_rigid_body_pdb_psf(
                rbt,
                resolve_file_path(block.input_pdb, self.config.source_path),
                resolve_file_path(block.input_psf, self.config.source_path),
                block.reference_point,
                system.get_particle_types(),
            )
        return rbt

    def _apply_tabulated(self, system) -> None:
        core = _core()
        registry = system.get_tables_registry()
        pairs = system.get_nonbonded_interactions()
        for pair in self.config.tabulated_pairs:
            path = self.resolved.tabulated_paths[pair.filename]
            index = registry.load_nonbonded(path)
            pairs.add_pair_nonbonded(
                core.PairNonBonded(pair.type_id_1, pair.type_id_2, Path(path).stem, index)
            )

    # -- staging payloads -----------------------------------------------------

    def build_particles(self) -> list:
        """One ``Particle`` per record, or per ``num`` when none were loaded.

        ``id`` is not set: ``SimManager::init`` assigns it from staging order.
        """
        core = _core()
        out = []
        if self.resolved.particles:
            for rec in self.resolved.particles:
                p = core.Particle()
                p.type_name = rec.type_name
                p.position = rec.position
                if rec.momentum is not None:
                    p.momentum = rec.momentum
                out.append(p)
            return out

        for block in self.config.particles:
            for _ in range(block.num):
                p = core.Particle()
                p.type_name = block.name
                out.append(p)
        return out

    def build_rigid_bodies(self) -> list:
        """``num`` instances per block, overwritten by inputRBCoordinates.

        ``type_id`` is not set: ``SystemState`` resolves it from ``type_name``.
        """
        core = _core()
        out = []
        for block in self.config.rigid_bodies:
            for _ in range(block.num):
                rb = core.RigidBody()
                rb.type_name = block.name
                rb.position = block.position
                if block.orientation is not None:
                    rb.orientation = block.orientation
                    rb.has_orientation = True
                rb.momentum = block.momentum
                rb.angular_momentum = block.angular_momentum
                rb.external_force = block.external_force
                rb.external_torque = block.external_torque
                out.append(rb)

        for i, coord in enumerate(self.resolved.rigid_body_coords):
            if i >= len(out):
                break
            out[i].position = coord.position
            out[i].orientation = coord.orientation
            out[i].has_orientation = True
        return out

    def build_bonded(self, system):
        """Every bonded term.

        An analytical name already carries its list position as
        ``function_index``; anything else is a table path, and the registry
        assigns its index as a side effect of loading it.
        """
        core = _core()
        bonded = core.BondedInteractions()
        topo = self.resolved.topology
        registry = system.get_tables_registry()
        source = self.config.source_path

        def index(rec, load) -> int:
            if rec.function_index >= 0:
                return rec.function_index
            return load(rec.function_name, source)

        for rec in topo.bonds:
            b = core.Bond(rec.ind1, rec.ind2, rec.function_name)
            b.function_index = index(rec, registry.get_or_load_bond)
            if rec.flag is not None:
                b.flag = getattr(core.BondFlag, rec.flag)
            bonded.add_bond(b)
        for rec in topo.angles:
            a = core.Angle(rec.ind1, rec.ind2, rec.ind3, rec.function_name)
            a.function_index = index(rec, registry.get_or_load_angle)
            bonded.add_angle(a)
        for rec in topo.dihedrals:
            d = core.Dihedral(
                rec.ind1, rec.ind2, rec.ind3, rec.ind4, rec.function_name
            )
            d.function_index = index(rec, registry.get_or_load_dihedral)
            bonded.add_dihedral(d)
        for rec in topo.exclusions:
            bonded.add_exclusion(core.Exclude(rec.ind1, rec.ind2))
        for rec in topo.restraints:
            bonded.add_restraint(core.Restraint(rec.ind, rec.r0, rec.k))
        return bonded


class BdSimulation:
    """One config, one ``SimSystem``, one ``SimManager``.

    Method names follow ``marsmodel.ArbdModel`` -- :meth:`prepare_for_simulation`
    then :meth:`simulate` -- so an arbdmodel script reads the same here.

    :param gpu: device index, spelled as in ``marsmodel``'s ``conf.gpu``; a
        list uses several devices. This selects a device, never a backend --
        the backend is fixed when the engine is built.

    :example:
        >>> sim = BdSimulation(parse_file("run.bd"), gpu=0)
        >>> sim.system.set_output_name("custom")
        >>> sim.simulate()
    """

    def __init__(self, config: BdConfig, gpu: int | list[int] = 0, *, resolved=None):
        core = _core()
        self.applier = BdApplier(config, resolved)
        self.system = core.SimSystem([gpu] if isinstance(gpu, int) else list(gpu))
        self.applier.apply_to(self.system)
        self._manager = None

    @property
    def config(self) -> BdConfig:
        return self.applier.config

    @property
    def manager(self):
        """The one manager bound to :attr:`system`, created on first use."""
        if self._manager is None:
            self._manager = _core().SimManager(self.system)
        return self._manager

    def prepare_for_simulation(self) -> None:
        """Stage particles, rigid bodies and topology, then initialize."""
        if not self.system.is_valid():
            raise ValueError("invalid system configuration")
        manager = self.manager
        particles = self.applier.build_particles()
        bodies = self.applier.build_rigid_bodies()
        fold_in_attached_particles(
            particles, bodies, {t.name: t for t in self.system.get_rigid_body_types()}
        )
        manager.stage_particles(particles)
        manager.stage_rigid_bodies(bodies)
        manager.stage_bonded_interactions(self.applier.build_bonded(self.system))
        manager.init()

    def simulate(self, output_name: str | None = None, *, steps: int | None = None) -> None:
        """Prepare and run.

        :param output_name: overrides the config's ``outputName``.
        :param steps: overrides the config's step count.
        """
        if output_name is not None:
            self.system.set_output_name(output_name)
        if steps is not None:
            self.system.set_num_steps(steps)
        self.prepare_for_simulation()
        self.manager.run()
