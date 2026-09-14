"""Phase A smoke tests for the compiled marsmd._core surface.

No fixture data, no GPU, no simulation. These assert that every core object is
importable, constructible, and that each bound property round-trips. Anything
that needs a device or a file belongs in a later phase.

Run with the repository root on PYTHONPATH:

    PYTHONPATH=$PWD python -m pytest Tests/interactive/python -v
"""

import numpy as np
import pytest

marsmd = pytest.importorskip("marsmd", reason="marsmd._core is not built")


# ---------------------------------------------------------------------------
# module surface
# ---------------------------------------------------------------------------

CORE_NAMES = [
    # objects
    "Particle", "ParticleType", "GridTerm", "RigidBody", "RigidBodyType",
    # bonded
    "Bond", "Angle", "Dihedral", "Exclude", "Restraint", "BondedInteractions",
    "RegisterPotential",
    # nonbonded
    "PairNonBonded", "LongRangeNonBonded", "NonBondedInteractions",
    # grids / tables
    "Grid", "GridKey", "GridManager", "Table", "TablesRegistry",
    # system / run
    "SimSystem", "ConfigParser", "SimManager", "PeriodicBox",
    "Temperature", "SimSteps",
    # enums
    "Periodicity", "DecomposerType", "DecomposeDirection", "LongRangeMethod",
    "OutputFormat", "TemperatureFormat", "IntegratorType",
    "BondFlag", "InteractionForm", "TabulatedType", "GridFormat", "GridType",
    "InterpolationOrder", "AnalyticalBondType", "AnalyticalAngleType",
    "AnalyticalDihedralType",
]


@pytest.mark.parametrize("name", CORE_NAMES)
def test_name_is_exported(name):
    assert hasattr(marsmd, name), f"marsmd.{name} missing from the compiled core"


def test_reexport_matches_core():
    """__init__.py lifts _core's public names; none may be dropped."""
    for name in dir(marsmd._core):
        if not name.startswith("_"):
            assert getattr(marsmd, name) is getattr(marsmd._core, name)


# ---------------------------------------------------------------------------
# particles
# ---------------------------------------------------------------------------


def test_particle_roundtrip():
    p = marsmd.Particle()
    p.type_name = "A"
    p.position = [1.0, 2.0, 3.0]
    p.momentum = [0.5, 0.0, -0.5]
    p.force = [0.0, 0.0, 0.0]
    p.energy = 1.25
    p.group_id = 7
    p.colvars_group_id = 3
    p.attached_rigid_body_id = 2

    assert p.type_name == "A"
    np.testing.assert_allclose(p.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(p.momentum, [0.5, 0.0, -0.5])
    assert p.energy == pytest.approx(1.25)
    assert p.group_id == 7
    assert p.colvars_group_id == 3
    assert p.attached_rigid_body_id == 2
    assert "A" in repr(p)


def test_bond_accepts_particles_or_indices():
    """Two ctors: particle handles (uids, resolved at staging) and raw indices."""
    a, b = marsmd.Particle(), marsmd.Particle()
    by_handle = marsmd.Bond(a, b, "tab/bond.dat")
    by_index = marsmd.Bond(0, 1, "tab/bond.dat")
    assert by_handle.name == by_index.name == "tab/bond.dat"
    assert (by_index.ind1, by_index.ind2) == (0, 1)


def test_particle_type_roundtrip():
    pt = marsmd.ParticleType("A")
    pt.mass = 12.0
    pt.charge = -1.0
    pt.radius = 2.5
    pt.eps = 0.1
    pt.mu = 0.0
    pt.num = 42
    pt.diffusivity = [43.5, 43.5, 43.5]
    pt.damping_coefficient = [10.0, 10.0, 10.0]

    assert pt.name == "A"
    assert pt.mass == pytest.approx(12.0)
    assert pt.charge == pytest.approx(-1.0)
    assert pt.num == 42
    np.testing.assert_allclose(pt.diffusivity, [43.5, 43.5, 43.5])
    np.testing.assert_allclose(pt.damping_coefficient, [10.0, 10.0, 10.0])


def test_particle_type_grid_name_fields():
    """assign_particle_type_ids() re-derives grid ids from these names."""
    pt = marsmd.ParticleType("A")
    pt.diffusion_grid_name = "../grids/diff.dx"
    pt.force_grid_names = ["fx.dx", "fy.dx", "fz.dx"]
    pt.pmf_grid_names = ["pmf-A.dx"]
    pt.rigid_body_potential_keys = ["rb-pot.dx"]

    assert pt.diffusion_grid_name == "../grids/diff.dx"
    assert list(pt.force_grid_names) == ["fx.dx", "fy.dx", "fz.dx"]
    assert list(pt.pmf_grid_names) == ["pmf-A.dx"]
    assert list(pt.rigid_body_potential_keys) == ["rb-pot.dx"]


def test_particle_type_force_grid_id_is_a_triple():
    pt = marsmd.ParticleType("A")
    np.testing.assert_array_equal(pt.force_grid_id, [-1, -1, -1])
    pt.force_grid_id = [3, 4, 5]
    np.testing.assert_array_equal(pt.force_grid_id, [3, 4, 5])


def test_grid_term():
    key = marsmd.GridKey()
    term = marsmd.GridTerm(key, scale=2.0, scale_slope=0.5, boundary_condition=1)
    assert term.scale == pytest.approx(2.0)
    assert term.scale_slope == pytest.approx(0.5)
    assert term.boundary_condition == 1


# ---------------------------------------------------------------------------
# rigid bodies
# ---------------------------------------------------------------------------


def test_rigid_body_roundtrip():
    rb = marsmd.RigidBody()
    rb.type_name = "nucleosome"
    rb.position = [10.0, 0.0, 0.0]
    rb.orientation = np.eye(3)
    rb.momentum = [1.0, 0.0, 0.0]
    rb.angular_momentum = [0.0, 1.0, 0.0]
    rb.external_force = [0.0, 0.0, -9.8]
    rb.external_torque = [0.0, 0.0, 1.0]
    rb.is_dummy = False
    rb.has_orientation = True

    assert rb.type_name == "nucleosome"
    np.testing.assert_allclose(rb.position, [10.0, 0.0, 0.0])
    np.testing.assert_allclose(rb.orientation, np.eye(3))
    np.testing.assert_allclose(rb.external_force, [0.0, 0.0, -9.8])
    np.testing.assert_allclose(rb.external_torque, [0.0, 0.0, 1.0])
    assert rb.has_orientation is True
    # Assigned by the applier when it lays out the attached block.
    assert rb.attached_start == -1
    assert rb.attached_count == 0


def test_rigid_body_type_roundtrip():
    rbt = marsmd.RigidBodyType("nucleosome")
    rbt.mass = 100.0
    rbt.moment_of_inertia = [1.0, 2.0, 3.0]
    rbt.damping_coefficient = [0.1, 0.1, 0.1]
    rbt.rotational_damping = [0.2, 0.2, 0.2]
    rbt.diffusivity = 5.0
    rbt.rotational_diffusivity = 0.5
    rbt.num_grid_files = 2

    assert rbt.name == "nucleosome"
    assert rbt.mass == pytest.approx(100.0)
    np.testing.assert_allclose(rbt.moment_of_inertia, [1.0, 2.0, 3.0])
    assert rbt.diffusivity == pytest.approx(5.0)
    assert rbt.num_grid_files == 2


def test_rigid_body_type_grid_vectors():
    rbt = marsmd.RigidBodyType("nucleosome")
    key = marsmd.GridKey()
    rbt.potential_grids = [marsmd.GridTerm(key, scale=1.0)]
    rbt.density_grids = [marsmd.GridTerm(key, scale=2.0)]
    rbt.pmf_grids = [marsmd.GridTerm(key, scale=3.0)]
    rbt.potential_grid_keys = ["pot.dx"]
    rbt.density_grid_keys = ["den.dx"]
    rbt.pmf_keys = ["pmf.dx"]

    assert len(rbt.potential_grids) == 1
    assert rbt.density_grids[0].scale == pytest.approx(2.0)
    assert rbt.pmf_grids[0].scale == pytest.approx(3.0)
    assert list(rbt.potential_grid_keys) == ["pot.dx"]
    assert list(rbt.density_grid_keys) == ["den.dx"]
    assert list(rbt.pmf_keys) == ["pmf.dx"]


def test_rigid_body_type_attached_particles():
    rbt = marsmd.RigidBodyType("nucleosome")
    p = marsmd.Particle()
    p.type_name = "A"
    rbt.attached_particles = [p]
    assert len(rbt.attached_particles) == 1
    assert rbt.attached_particles[0].type_name == "A"


# ---------------------------------------------------------------------------
# bonded interactions
# ---------------------------------------------------------------------------


def test_bond_index_ctor():
    b = marsmd.Bond(3, 7, "tab/bond.dat")
    assert (b.ind1, b.ind2) == (3, 7)
    assert b.name == "tab/bond.dat"
    # -1 until TablesRegistry resolves it; the device reads it raw.
    assert b.function_index == -1
    b.function_index = 2
    assert b.function_index == 2


def test_angle_and_dihedral_index_ctors():
    a = marsmd.Angle(1, 2, 3, "tab/angle.dat")
    assert (a.ind1, a.ind2, a.ind3) == (1, 2, 3)
    assert a.form == marsmd.InteractionForm.Tabulated

    d = marsmd.Dihedral(1, 2, 3, 4, "tab/dihedral.dat")
    assert (d.ind1, d.ind2, d.ind3, d.ind4) == (1, 2, 3, 4)
    assert d.form == marsmd.InteractionForm.Tabulated


def test_exclude_index_ctor_and_ordering():
    e = marsmd.Exclude(2, 5)
    assert (e.ind1, e.ind2) == (2, 5)
    assert e == marsmd.Exclude(2, 5)
    assert e != marsmd.Exclude(2, 6)
    assert marsmd.Exclude(1, 2) < marsmd.Exclude(1, 3)


def test_restraint_index_ctor():
    r = marsmd.Restraint(4, [1.0, 2.0, 3.0], 10.0)
    assert r.ind == 4
    np.testing.assert_allclose(r.r0, [1.0, 2.0, 3.0])
    assert r.k == pytest.approx(10.0)


def test_bond_add_exclusion_is_gone():
    """It appended to a caller-supplied vector that nanobind copied -- a no-op."""
    assert not hasattr(marsmd.Bond, "add_exclusion")


def test_bonded_interactions_roundtrip():
    bi = marsmd.BondedInteractions()
    bi.add_bond(marsmd.Bond(0, 1, "b.dat"))
    bi.add_bond(marsmd.Bond(1, 2, "b.dat"))
    bi.add_angle(marsmd.Angle(0, 1, 2, "a.dat"))
    bi.add_dihedral(marsmd.Dihedral(0, 1, 2, 3, "d.dat"))
    bi.add_exclusion(marsmd.Exclude(0, 2))
    bi.add_restraint(marsmd.Restraint(0, [0.0, 0.0, 0.0], 1.0))

    assert bi.get_num_bonds() == 2
    assert bi.get_num_angles() == 1
    assert bi.get_num_dihedrals() == 1

    assert [(b.ind1, b.ind2) for b in bi.get_bonds()] == [(0, 1), (1, 2)]
    assert len(bi.get_angles()) == 1
    assert len(bi.get_dihedrals()) == 1
    assert len(bi.get_exclusions()) == 1
    assert len(bi.get_restraints()) == 1

    bi.clear()
    assert bi.get_num_bonds() == 0


def test_bonded_getters_return_copies():
    bi = marsmd.BondedInteractions()
    bi.add_bond(marsmd.Bond(0, 1, "b.dat"))
    snapshot = bi.get_bonds()
    snapshot.append(marsmd.Bond(9, 9, "b.dat"))
    assert bi.get_num_bonds() == 1


# ---------------------------------------------------------------------------
# nonbonded interactions -- pynonbonded.cpp was orphaned before Phase A
# ---------------------------------------------------------------------------


def test_pair_nonbonded():
    a, b = marsmd.ParticleType("A"), marsmd.ParticleType("B")
    pair = marsmd.PairNonBonded(a, b, "tab/AB.dat")
    assert pair.type_name_1 == "A"
    assert pair.type_name_2 == "B"
    assert pair.name == "tab/AB.dat"


def test_nonbonded_interactions_container():
    a, b = marsmd.ParticleType("A"), marsmd.ParticleType("B")
    nb_ = marsmd.NonBondedInteractions()
    nb_.add_pair_nonbonded(marsmd.PairNonBonded(a, b, "tab/AB.dat"))
    lr = marsmd.LongRangeNonBonded()
    lr.name = "debye"
    nb_.add_long_range_nonbonded(lr)

    assert nb_.get_num_pair_nonbonded() == 1
    assert nb_.get_num_long_range_nonbonded() == 1


# ---------------------------------------------------------------------------
# grids and tables
# ---------------------------------------------------------------------------


def test_grid_from_numpy_roundtrip():
    values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    grid = marsmd.Grid.from_numpy(
        values, origin=[0.0, 0.0, 0.0], spacing=[1.0, 1.0, 1.0]
    )
    assert (grid.nx(), grid.ny(), grid.nz()) == (2, 3, 4)
    np.testing.assert_allclose(grid.to_numpy(), values)


def test_grid_manager_is_empty_by_default():
    gm = marsmd.GridManager()
    assert gm.num_grids() == 0
    assert not gm.has_grid("nope.dx")


def test_table_set_values():
    t = marsmd.Table(marsmd.TabulatedType.Bond)
    t.name = "tab/bond.dat"
    t.set_values([0.0, 1.0, 4.0, 9.0], start=0.0, step_size=1.0)
    assert t.name == "tab/bond.dat"
    assert t.type == marsmd.TabulatedType.Bond
    np.testing.assert_allclose(list(t.Y), [0.0, 1.0, 4.0, 9.0])
    np.testing.assert_allclose(list(t.X), [0.0, 1.0, 2.0, 3.0])


def test_tables_registry_starts_empty():
    reg = marsmd.TablesRegistry()
    assert reg.get_bond_name_to_idx() == {}
    assert reg.get_angle_name_to_idx() == {}
    assert reg.get_dihedral_name_to_idx() == {}
    assert "TablesRegistry" in repr(reg)


# ---------------------------------------------------------------------------
# SimSystem
# ---------------------------------------------------------------------------


@pytest.fixture
def system():
    """Device 0 of whichever backend this engine was built for."""
    return marsmd.SimSystem([0])


def test_sim_system_scalars(system):
    sys = system
    sys.set_temperature_value(295.0)
    sys.set_cutoff(12.0)
    sys.set_timestep(0.02)
    sys.set_num_steps(1000)
    sys.set_output_period(100)
    sys.set_output_name("out/run")

    assert sys.get_temperature() == pytest.approx(295.0)
    assert sys.get_cutoff() == pytest.approx(12.0)
    assert sys.get_timestep() == pytest.approx(0.02)
    assert sys.get_num_steps() == 1000
    assert sys.get_output_period() == pytest.approx(100)
    assert sys.get_output_name() == "out/run"


def test_sim_system_box(system):
    sys = system
    sys.set_box_size(100.0, 200.0, 300.0)
    sys.set_origin(-50.0, -100.0, -150.0)
    sys.set_periodicity(True, True, False)
    np.testing.assert_allclose(sys.get_box_size(), [100.0, 200.0, 300.0])


def test_integrator_type_setters_are_callable(system):
    """The enum was unbound, so these four setters could not be called at all."""
    sys = system
    sys.set_particle_integrator_type(marsmd.IntegratorType.Brownian)
    assert sys.get_particle_algorithm() == marsmd.IntegratorType.Brownian

    sys.set_rigid_body_integrator_type(marsmd.IntegratorType.Langevin)
    assert sys.get_rigid_body_algorithm() == marsmd.IntegratorType.Langevin


def test_newly_bound_sim_system_setters(system):
    sys = system
    sys.set_base_seed(12345)
    sys.set_pairlist_cutoff(15.0)
    sys.set_salt_concentration(0.15)
    sys.set_neighbor_list_rebuild_period(20.0)
    sys.set_reorder_period(50)
    sys.set_rb_update_period(10)
    sys.set_estimated_particles(100_000)
    assert sys.get_neighbor_list_rebuild_period() == pytest.approx(20.0)


def test_particle_type_registration(system):
    sys = system
    pt = marsmd.ParticleType("A")
    pt.mass = 12.0
    sys.add_particle_type(pt)
    types = sys.get_particle_types()
    assert len(types) == 1
    assert types[0].name == "A"


def test_get_particle_types_returns_a_snapshot(system):
    sys = system
    sys.add_particle_type(marsmd.ParticleType("A"))
    snapshot = sys.get_particle_types()
    snapshot.append(marsmd.ParticleType("B"))
    assert len(sys.get_particle_types()) == 1


def test_sim_system_owns_grid_manager_and_tables(system):
    sys = system
    assert sys.get_grid_manager().num_grids() == 0
    # reference_internal: TablesRegistry holds device buffers and must not be copied.
    assert sys.get_tables_registry().get_bond_name_to_idx() == {}
    assert sys.get_nonbonded_interactions().get_num_pair_nonbonded() == 0


def test_sim_system_enums_round_trip(system):
    sys = system
    sys.set_long_range_method(marsmd.LongRangeMethod.Direct)
    assert sys.get_long_range_method() == marsmd.LongRangeMethod.Direct
    sys.set_output_format(marsmd.OutputFormat.DCD)
    assert sys.get_output_format() == marsmd.OutputFormat.DCD
    sys.set_decomposer_type(marsmd.DecomposerType.Spatial, marsmd.DecomposeDirection.Z)
    assert sys.get_decomposer_type() == marsmd.DecomposerType.Spatial


# ---------------------------------------------------------------------------
# SimManager -- construction and staging only, no init()/run()
# ---------------------------------------------------------------------------


def test_sim_manager_staging(system):
    sys = system
    sys.add_particle_type(marsmd.ParticleType("A"))

    particles = []
    for i in range(4):
        p = marsmd.Particle()
        p.type_name = "A"
        p.position = [float(i), 0.0, 0.0]
        particles.append(p)

    bi = marsmd.BondedInteractions()
    bi.add_bond(marsmd.Bond(0, 1, "b.dat"))

    mgr = marsmd.SimManager(sys)
    mgr.stage_particles(particles)
    mgr.stage_bonded_interactions(bi)
    mgr.stage_rigid_bodies([])
    assert "SimManager" in repr(mgr)
