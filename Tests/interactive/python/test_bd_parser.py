"""Phase C tests for the pure-Python .bd parser.

No compiled extension, no GPU, no fixture data: every input is inline text or a
tmp_path file. That is the point of keeping the parser free of ``_core`` -- a
grammar regression is catchable without a build.

    PYTHONPATH=$PWD python -m pytest Tests/interactive/python/test_bd_parser.py -v
"""

from __future__ import annotations

import json
import textwrap

import pytest

from marsmd.bd import BdKeywordError, BdParseError, BdParser, parse_string
from marsmd.bd import aux, paths
from marsmd.bd.model import BdConfig
from marsmd.bd.tokens import (
    is_comment_or_blank,
    parse_int,
    parse_matrix3_rows,
    split_key_value,
    strip_trailing_comment,
)


def bd(text: str) -> BdConfig:
    """Parse dedented inline .bd text."""
    return parse_string(textwrap.dedent(text), source_path="run.bd")


# ===========================================================================
# paths -- port of resolve_file_path (src/Header.h:64-90)
# ===========================================================================


@pytest.mark.parametrize(
    "file_path,config_path,expected",
    [
        # Absolute wins outright.
        ("/abs/grid.dx", "sim/run.bd", "/abs/grid.dx"),
        ("/abs/grid.dx", "", "/abs/grid.dx"),
        # Relative resolves against the config's directory, unnormalized.
        ("grid.dx", "sim/run.bd", "sim/grid.dx"),
        ("../grids/grid-P.dx", "sim/run.bd", "sim/../grids/grid-P.dx"),
        ("potentials/null.dx", "a/b/c/run.bd", "a/b/c/potentials/null.dx"),
        # No separator in the config path means the current directory.
        ("grid.dx", "run.bd", "./grid.dx"),
        ("grid.dx", "", "./grid.dx"),
        # Trailing separator is kept as-is.
        ("grid.dx", "sim/", "sim/grid.dx"),
    ],
)
def test_resolve_file_path(file_path, config_path, expected):
    assert paths.resolve_file_path(file_path, config_path) == expected


def test_resolve_file_path_expands_home(monkeypatch):
    monkeypatch.setenv("HOME", "/home/someone")
    assert paths.resolve_file_path("~/g.dx", "run.bd") == "/home/someone/g.dx"


def test_resolve_file_path_without_home_returns_input(monkeypatch):
    """C++ falls back to the raw path when HOME is unset."""
    monkeypatch.delenv("HOME", raising=False)
    assert paths.resolve_file_path("~/g.dx", "run.bd") == "~/g.dx"


def test_resolve_file_path_does_not_normalize():
    """`..` must survive: the string is the GridManager map key."""
    resolved = paths.resolve_file_path("../grids/grid-P.dx", "sim/run.bd")
    assert ".." in resolved


# ===========================================================================
# tokens
# ===========================================================================


@pytest.mark.parametrize(
    "line,expected",
    [
        ("", True),
        ("   ", True),
        ("\t\n", True),
        ("# comment", True),
        ("   # indented comment", True),
        ("## double", True),
        ("temperature 295", False),
        ("  temperature 295", False),
        ("temperature 295 # trailing", False),
    ],
)
def test_is_comment_or_blank(line, expected):
    assert is_comment_or_blank(line) is expected


def test_split_key_value_collapses_whitespace():
    """Reader::parseLine re-joins tokens with single spaces."""
    assert split_key_value("tabulatedPotential  1") == ("tabulatedPotential", "1")
    assert split_key_value("inertia  1.0   2.0\t3.0") == ("inertia", "1.0 2.0 3.0")
    assert split_key_value("   ") == ("", "")


def test_strip_trailing_comment():
    assert strip_trailing_comment("0 # deprecated") == "0"
    assert strip_trailing_comment("0") == "0"
    assert strip_trailing_comment("a b # c # d") == "a b"


def test_parse_int_stops_at_first_non_digit_like_stoi():
    """`0 # deprecated` must read as 0 without stripping comments first."""
    line = type("L", (), {"line_no": 1, "raw": ""})()
    assert parse_int("0 # deprecated", key="k", line=line) == 0
    assert parse_int("12abc", key="k", line=line) == 12
    assert parse_int("-5", key="k", line=line) == -5
    assert parse_int("+7", key="k", line=line) == 7
    with pytest.raises(BdParseError):
        parse_int("abc", key="k", line=line)


def test_parse_matrix3_rows_is_row_major():
    m = parse_matrix3_rows("1 2 3 4 5 6 7 8 9")
    assert m == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_parse_matrix3_rows_falls_back_to_identity():
    assert parse_matrix3_rows("1 2 3") == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# ===========================================================================
# globals
# ===========================================================================


def test_defaults_match_apply_defaults():
    """A config with no keys must land on the engine's own defaults."""
    g = bd("").globals
    assert g.temperature == pytest.approx(298.15)
    assert g.cutoff == pytest.approx(10.0)
    assert g.timestep == pytest.approx(1e-5)
    assert g.steps == 1000
    assert g.pairlist_rebuild_period == pytest.approx(1000.0)
    assert g.output_period == pytest.approx(10.0)
    assert g.output_energy_period == pytest.approx(100.0)
    assert g.output_name == "out"


def test_global_scalars():
    g = bd(
        """
        seed 70650
        timestep 2e-05
        steps 100000
        temperature 295
        cutoff 35
        pairlistDistance 10
        decompPeriod 40
        outputPeriod 1000.0
        outputEnergyPeriod 1000.0
        outputName out/run
        """
    ).globals
    assert g.seed == 70650
    assert g.timestep == pytest.approx(2e-5)
    assert g.steps == 100000
    assert g.temperature == pytest.approx(295.0)
    assert g.cutoff == pytest.approx(35.0)
    assert g.pairlist_distance == pytest.approx(10.0)
    assert g.pairlist_rebuild_period == pytest.approx(40.0)
    assert g.output_name == "out/run"


def test_pairlist_cutoff_is_cutoff_plus_skin():
    """pairlistDistance is a skin, not the cutoff itself."""
    g = bd("cutoff 35\npairlistDistance 10\n").globals
    assert g.pairlist_cutoff == pytest.approx(45.0)


def test_pairlist_cutoff_defaults_to_cutoff_with_zero_skin():
    g = bd("cutoff 35\n").globals
    assert g.pairlist_cutoff == pytest.approx(35.0)


@pytest.mark.parametrize(
    "text,field,expected",
    [
        ("ParticleDynamicType Langevin", "particle_dynamic_type", "Langevin"),
        ("particleDynamicType brownian", "particle_dynamic_type", "Brownian"),
        ("algorithm VELOCITYVERLET", "particle_dynamic_type", "VelocityVerlet"),
        ("RigidBodyDynamicType Langevin", "rigid_body_dynamic_type", "Langevin"),
        ("rigidBodyAlgorithm Brownian", "rigid_body_dynamic_type", "Brownian"),
        ("outputFormat dcd", "output_format", "DCD"),
        ("outputFormat DCD", "output_format", "DCD"),
        ("decomposer spatial", "decomposer", "Spatial"),
        ("longRangeMethod none", "long_range_method", "None"),
    ],
)
def test_enum_keys_are_case_insensitive(text, field, expected):
    assert getattr(bd(text).globals, field) == expected


def test_unknown_enum_value_becomes_none_not_an_error():
    """The engine warns and keeps its default rather than failing."""
    assert bd("outputFormat xyz").globals.output_format is None


def test_vector_globals():
    g = bd(
        """
        origin -1500.0 -1500.0 -1500.0
        systemSize 3000 3000 3000
        """
    ).globals
    assert g.origin == (-1500.0, -1500.0, -1500.0)
    assert g.system_size == (3000.0, 3000.0, 3000.0)


def test_snake_case_variants_are_accepted():
    g = bd(
        """
        output_name run2
        output_period 5
        system_size 10 10 10
        pairlist_distance 2
        """
    ).globals
    assert g.output_name == "run2"
    assert g.output_period == pytest.approx(5.0)
    assert g.system_size == (10.0, 10.0, 10.0)
    assert g.pairlist_distance == pytest.approx(2.0)


# ===========================================================================
# particle blocks
# ===========================================================================


def test_particle_block_fields():
    config = bd(
        """
        particle B
        num 500
        mass 181
        transDamping 1239.8 1239.8 1239.8
        gridFile potentials/null.dx
        rigidBodyPotential Bbead
        """
    )
    assert len(config.particles) == 1
    block = config.particles[0]
    assert block.name == "B"
    assert block.num == 500
    assert block.mass == pytest.approx(181.0)
    assert block.trans_damping == pytest.approx((1239.8, 1239.8, 1239.8))
    assert [g.filename for g in block.grid_files] == ["potentials/null.dx"]
    assert block.rigid_body_potential_keys == ["Bbead"]


def test_particle_blocks_are_ordered_and_indexable():
    """Block order is the type id tabulatedFile i@j refers to."""
    config = bd("particle B\nnum 1\nparticle P\nnum 2\nparticle rb\nnum 0\n")
    assert [b.name for b in config.particles] == ["B", "P", "rb"]
    assert config.particle_type_index("B") == 0
    assert config.particle_type_index("P") == 1
    assert config.particle_type_index("nope") == -1


def test_diffusion_and_damping_broadcast_a_single_value():
    """One value fills x/y/z -- ConfigParser.cpp:683-686."""
    block = bd("particle A\ndiffusion 43.5\ntransDamping 10\n").particles[0]
    assert block.diffusion == pytest.approx((43.5, 43.5, 43.5))
    assert block.trans_damping == pytest.approx((10.0, 10.0, 10.0))


def test_grid_file_accepts_several_files_on_one_line():
    block = bd("particle A\ngridFile a.dx b.dx c.dx\n").particles[0]
    assert [g.filename for g in block.grid_files] == ["a.dx", "b.dx", "c.dx"]


def test_grid_file_scale_broadcasts_one_value():
    block = bd("particle A\ngridFile a.dx b.dx\ngridFileScale 2.0\n").particles[0]
    assert [g.scale for g in block.grid_files] == [2.0, 2.0]


def test_grid_file_scale_matches_one_to_one():
    block = bd("particle A\ngridFile a.dx b.dx\ngridFileScale 2.0 3.0\n").particles[0]
    assert [g.scale for g in block.grid_files] == [2.0, 3.0]


def test_grid_file_scale_with_wrong_arity_is_ignored():
    """Neither 1 nor N values: the engine warns and leaves the grids alone."""
    block = bd(
        "particle A\ngridFile a.dx b.dx c.dx\ngridFileScale 2.0 3.0\n"
    ).particles[0]
    assert [g.scale for g in block.grid_files] == [1.0, 1.0, 1.0]


def test_grid_file_scale_may_precede_the_grid_it_scales():
    """These keys are deferred to the end of the block on purpose."""
    before = bd("particle A\ngridFileScale 2.0\ngridFile a.dx\n").particles[0]
    after = bd("particle A\ngridFile a.dx\ngridFileScale 2.0\n").particles[0]
    assert [g.scale for g in before.grid_files] == [2.0]
    assert before.grid_files == after.grid_files


def test_grid_file_scale_without_grids_is_dropped():
    block = bd("particle A\ngridFileScale 2.0\n").particles[0]
    assert block.grid_files == []


def test_grid_file_boundary_conditions():
    block = bd(
        "particle A\ngridFile a.dx b.dx c.dx\n"
        "gridFileBoundaryConditions dirichlet neumann periodic\n"
    ).particles[0]
    assert [g.boundary_condition for g in block.grid_files] == [0, 1, 2]


def test_unrecognized_boundary_condition_keeps_the_grids_own():
    block = bd(
        "particle A\ngridFile a.dx\ngridFileBoundaryConditions nonsense\n"
    ).particles[0]
    assert block.grid_files[0].boundary_condition == -1


def test_force_grid_files_need_exactly_three():
    ok = bd("particle A\nforceGridFiles x.dx y.dx z.dx\n").particles[0]
    assert ok.force_grid_files == ["x.dx", "y.dx", "z.dx"]
    bad = bd("particle A\nforceGridFiles x.dx y.dx\n").particles[0]
    assert bad.force_grid_files == []


def test_force_grid_scale_broadcasts():
    block = bd("particle A\nforceGridScale 2.0\n").particles[0]
    assert block.force_grid_scale == pytest.approx((2.0, 2.0, 2.0))


def test_grid_file_smd_and_diffusion_grid():
    block = bd(
        "particle A\ngridFileSMD 100\ndiffusionGridFile ../grids/diff.dx\n"
    ).particles[0]
    assert block.grid_file_smd == 100
    assert block.diffusion_grid_file == "../grids/diff.dx"


# ===========================================================================
# rigidBody blocks
# ===========================================================================


def test_rigid_body_block_fields():
    rb = bd(
        """
        rigidBody SSB
        num 14
        mass 48856.41
        inertia 19375322.0 18923208.0 13511100.0
        transDamping 779.07 757.10 735.26
        rotDamping 3096.68 3027.72 3547.02
        """
    ).rigid_bodies[0]
    assert rb.name == "SSB"
    assert rb.num == 14
    assert rb.mass == pytest.approx(48856.41)
    assert rb.inertia == pytest.approx((19375322.0, 18923208.0, 13511100.0))
    assert rb.rot_damping == pytest.approx((3096.68, 3027.72, 3547.02))


def test_rigid_body_grid_takes_key_and_file():
    rb = bd(
        """
        rigidBody SSB
        potentialGrid Pbead ../grids/grid-P.dx
        potentialGrid elec ../grids/1eyg.elec.dx
        """
    ).rigid_bodies[0]
    assert [(g.key, g.filename) for g in rb.potential_grids] == [
        ("Pbead", "../grids/grid-P.dx"),
        ("elec", "../grids/1eyg.elec.dx"),
    ]


def test_bare_rigid_body_grid_file_doubles_as_its_key():
    rb = bd("rigidBody R\npotentialGrid only.dx\n").rigid_bodies[0]
    assert rb.potential_grids[0].key == "only.dx"
    assert rb.potential_grids[0].filename == "only.dx"


def test_rigid_body_grid_scale_is_keyed():
    """Each scale line names the grid it applies to."""
    rb = bd(
        """
        rigidBody SSB
        potentialGrid Pbead p.dx
        potentialGridScale Pbead 0.578277
        potentialGrid vdw0 v.dx
        potentialGrid elec e.dx
        potentialGridScale elec 0.25
        """
    ).rigid_bodies[0]
    by_key = {g.key: g.scale for g in rb.potential_grids}
    assert by_key["Pbead"] == pytest.approx(0.578277)
    assert by_key["elec"] == pytest.approx(0.25)
    assert by_key["vdw0"] == pytest.approx(1.0)


def test_bare_rigid_body_scale_applies_to_every_grid_of_that_kind():
    rb = bd(
        "rigidBody R\npotentialGrid a a.dx\npotentialGrid b b.dx\n"
        "potentialGridScale 3.0\n"
    ).rigid_bodies[0]
    assert [g.scale for g in rb.potential_grids] == [3.0, 3.0]


def test_rigid_body_scale_for_an_unknown_key_is_ignored():
    rb = bd(
        "rigidBody R\npotentialGrid a a.dx\npotentialGridScale nope 3.0\n"
    ).rigid_bodies[0]
    assert rb.potential_grids[0].scale == pytest.approx(1.0)


def test_rigid_body_grid_kinds_scale_independently():
    rb = bd(
        """
        rigidBody R
        potentialGrid k p.dx
        densityGrid k d.dx
        pmf k m.dx
        potentialGridScale k 2.0
        densityGridScale k 3.0
        pmfScale k 4.0
        """
    ).rigid_bodies[0]
    assert rb.potential_grids[0].scale == pytest.approx(2.0)
    assert rb.density_grids[0].scale == pytest.approx(3.0)
    assert rb.pmf_grids[0].scale == pytest.approx(4.0)


def test_rigid_body_orientation_is_row_major_and_sets_the_flag():
    rb = bd("rigidBody R\norientation 1 0 0 0 1 0 0 0 1\n").rigid_bodies[0]
    assert rb.orientation == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert rb.has_orientation is True


def test_rigid_body_without_orientation_reports_no_orientation():
    assert bd("rigidBody R\n").rigid_bodies[0].has_orientation is False


def test_rigid_body_per_instance_loads():
    rb = bd(
        """
        rigidBody R
        position 1 2 3
        momentum 4 5 6
        angularMomentum 7 8 9
        constantForce 0 0 -9.8
        constantTorque 0 0 1
        """
    ).rigid_bodies[0]
    assert rb.position == (1.0, 2.0, 3.0)
    assert rb.momentum == (4.0, 5.0, 6.0)
    assert rb.angular_momentum == (7.0, 8.0, 9.0)
    assert rb.external_force == pytest.approx((0.0, 0.0, -9.8))
    assert rb.external_torque == (0.0, 0.0, 1.0)


def test_negative_rigid_body_num_is_an_error():
    with pytest.raises(BdParseError, match="num must be >= 0"):
        bd("rigidBody R\nnum -1\n")


def test_pdb_without_psf_is_an_error():
    with pytest.raises(BdParseError, match="must be given together"):
        bd("rigidBody R\ninputPdb t.pdb\n")


def test_pdb_and_psf_together_are_recorded_unresolved():
    rb = bd(
        """
        rigidBody R
        inputPdb ssb_template.pdb
        inputPsf ssb_template.psf
        referencePoint 0.0 0.0 0.0
        """
    ).rigid_bodies[0]
    assert rb.input_pdb == "ssb_template.pdb"
    assert rb.input_psf == "ssb_template.psf"
    assert rb.reference_point == (0.0, 0.0, 0.0)


# ===========================================================================
# block cursor semantics
# ===========================================================================


def test_a_block_ends_at_the_next_block_header():
    config = bd("particle A\nnum 1\nparticle B\nnum 2\n")
    assert [b.num for b in config.particles] == [1, 2]


def test_top_level_keys_between_blocks_do_not_close_the_block():
    """Legacy persistent-cursor semantics, a superset of get_elements.

    get_elements terminates a block at the first unrecognized key and drops
    what follows. Here the key is handled at top level and the block stays
    open, so a field appearing after it is still attributed correctly.
    """
    config = bd(
        """
        particle A
        num 1
        inputParticles run.particles.txt
        mass 42
        """
    )
    assert config.input_particles == "run.particles.txt"
    assert config.particles[0].mass == pytest.approx(42.0)


def test_rigid_body_block_survives_an_interleaved_top_level_key():
    config = bd(
        """
        rigidBody R
        num 2
        inputRBCoordinates run.rbcoords.txt
        mass 7
        """
    )
    assert config.input_rb_coordinates == "run.rbcoords.txt"
    assert config.rigid_bodies[0].mass == pytest.approx(7.0)


def test_mass_is_attributed_to_whichever_block_is_open():
    config = bd("particle A\nmass 1\nrigidBody R\nmass 2\n")
    assert config.particles[0].mass == pytest.approx(1.0)
    assert config.rigid_bodies[0].mass == pytest.approx(2.0)


# ===========================================================================
# top-level element keys
# ===========================================================================


def test_tabulated_file_pairs():
    config = bd(
        """
        tabulatedFile 0@0@potentials/run-nb.B-B.dat
        tabulatedFile 0@1@potentials/run-nb.B-P.dat
        tabulatedFile 1@1@potentials/run-nb.P-P.dat
        """
    )
    assert [
        (p.type_id_1, p.type_id_2, p.filename) for p in config.tabulated_pairs
    ] == [
        (0, 0, "potentials/run-nb.B-B.dat"),
        (0, 1, "potentials/run-nb.B-P.dat"),
        (1, 1, "potentials/run-nb.P-P.dat"),
    ]


def test_tabulated_file_path_may_contain_an_at_sign():
    """Only the first two @ separate; the rest is the path."""
    pair = bd("tabulatedFile 1@0@a@b.dat").tabulated_pairs[0]
    assert (pair.type_id_1, pair.type_id_2, pair.filename) == (1, 0, "a@b.dat")


def test_descending_tabulated_file_indices_are_preserved():
    """`1@0@...` must not be silently reordered to 0@1."""
    pair = bd("tabulatedFile 1@0@p.dat").tabulated_pairs[0]
    assert (pair.type_id_1, pair.type_id_2) == (1, 0)


def test_malformed_tabulated_file_raises_in_strict_mode():
    with pytest.raises(BdParseError, match="tabulatedFile"):
        bd("tabulatedFile 0@nofile")


def test_malformed_tabulated_file_is_collected_when_not_strict():
    config = parse_string("tabulatedFile 0@nofile\n", unknown="ignore")
    assert config.tabulated_pairs == []
    assert any("tabulatedFile" in u.reason for u in config.unsupported)


def test_topology_files_keep_their_order_and_kind():
    config = bd(
        """
        inputBonds potentials/run.bond.txt
        inputAngles potentials/run.angle.txt
        inputDihedrals potentials/run.dihedral.txt
        inputExcludes potentials/run.exclusion.txt
        """
    )
    assert [(t.kind, t.filename) for t in config.topology_files] == [
        ("inputBonds", "potentials/run.bond.txt"),
        ("inputAngles", "potentials/run.angle.txt"),
        ("inputDihedrals", "potentials/run.dihedral.txt"),
        ("inputExcludes", "potentials/run.exclusion.txt"),
    ]


def test_explicit_particle_source_detection():
    assert bd("inputParticles p.txt").has_explicit_particle_source is True
    assert bd("restartCoordinates r.txt").has_explicit_particle_source is True
    assert bd("particle A\nnum 5\n").has_explicit_particle_source is False


def test_rigid_body_grid_grid_period():
    assert bd("rigidBodyGridGridPeriod 20").globals.rb_grid_grid_period == 20


# ===========================================================================
# unknown-keyword policy
# ===========================================================================


def test_strict_is_the_default_and_raises():
    with pytest.raises(BdKeywordError, match="wubbleflurb"):
        bd("wubbleflurb 3")


def test_warn_policy_warns_and_collects():
    with pytest.warns(UserWarning, match="wubbleflurb"):
        config = parse_string("wubbleflurb 3\n", unknown="warn")
    assert [u.key for u in config.unsupported] == ["wubbleflurb"]


def test_ignore_policy_is_silent():
    config = parse_string("wubbleflurb 3\n", unknown="ignore")
    assert [u.key for u in config.unsupported] == ["wubbleflurb"]


def test_known_but_unapplied_keys_never_trip_strict_mode():
    """These appear all over the existing fixtures."""
    config = bd(
        """
        numberFluct 0
        interparticleForce 1
        fullLongRange 0
        tabulatedPotential  1
        tabulatedBondFile /abs/BPP.dat
        tabulatedAngleFile /abs/b1p2b2.dat
        tabulatedDihedralFile /abs/b1p2p3b3.dat
        """
    )
    assert {u.key for u in config.unsupported} == {
        "numberFluct",
        "interparticleForce",
        "fullLongRange",
        "tabulatedPotential",
        "tabulatedBondFile",
        "tabulatedAngleFile",
        "tabulatedDihedralFile",
    }
    assert all(u.reason for u in config.unsupported)


def test_unsupported_keys_record_their_line_number():
    config = bd("\n\nnumberFluct 0\n")
    assert config.unsupported[0].line_no == 3


def test_attached_particles_is_recorded_not_applied():
    """Not in rigid_body_field_keys, so the engine drops it -- as do we."""
    config = bd("rigidBody R\nattachedParticles run.attached.txt\n")
    assert [u.key for u in config.unsupported] == ["attachedParticles"]


@pytest.mark.parametrize("key", ["pmf", "pmfFile", "gridFile"])
def test_rigid_body_pmf_key_aliases(key):
    """v1 spelled the PMF list ``gridFile`` here; all three are one list.

    Live case: Tests/integration/rb-dipole/BrownDyn.v1.bd:34.
    """
    config = bd(f"rigidBody point\ndensityGrid elec dipole.dx\n{key} elec uniform.dx\n")
    rb = config.rigid_bodies[0]
    assert [g.filename for g in rb.density_grids] == ["dipole.dx"]
    assert [(g.key, g.filename) for g in rb.pmf_grids] == [("elec", "uniform.dx")]
    assert config.unsupported == []


def test_rigid_body_pmf_aliases_share_one_list_and_one_scale():
    """pmfScale indexes the merged list, so aliases must not fork it."""
    config = bd(
        "rigidBody point\n"
        "pmf a a.dx\n"
        "gridFile b b.dx\n"
        "pmfFile c c.dx\n"
        "pmfScale b 2.5\n"
    )
    rb = config.rigid_bodies[0]
    assert [g.key for g in rb.pmf_grids] == ["a", "b", "c"]
    assert [g.scale for g in rb.pmf_grids] == [1.0, 2.5, 1.0]


def test_grid_file_in_a_particle_block_is_still_a_particle_pmf():
    """The alias is rigidBody-only; the particle block keeps its own meaning."""
    config = bd("particle A\nnum 1\ngridFile p.dx\n")
    assert [g.filename for g in config.particles[0].grid_files] == ["p.dx"]
    assert config.unsupported == []


def test_rigid_body_key_inside_a_particle_block_is_recorded():
    config = bd("particle A\nnum 1\ninertia 1 2 3\n")
    assert [u.key for u in config.unsupported] == ["inertia"]
    assert "rigidBody key" in config.unsupported[0].reason


def test_a_misplaced_block_key_never_fails_strict_mode():
    """A block break is not a typo; strict mode is for typos."""
    config = parse_string("rigidBody R\ndiffusion 1 2 3\n")  # strict by default
    assert config.unsupported[0].key == "diffusion"


def test_a_shared_key_is_not_treated_as_misplaced():
    """mass and num belong to both block kinds."""
    config = bd("rigidBody R\nmass 5\nnum 2\n")
    assert config.unsupported == []
    assert config.rigid_bodies[0].mass == pytest.approx(5.0)


def test_invalid_policy_name_is_rejected():
    with pytest.raises(ValueError, match="strict/warn/ignore"):
        BdParser(unknown="loud")


# ===========================================================================
# comments
# ===========================================================================


def test_trailing_comments_survive_integer_parsing():
    """`numberFluct 0 # deprecated` must not need comment stripping."""
    config = bd("steps 100 # was 1000\n")
    assert config.globals.steps == 100


def test_comment_stripping_is_opt_in():
    off = parse_string("outputName out # trailing\n")
    on = parse_string("outputName out # trailing\n", strip_comments=True)
    assert off.globals.output_name == "out # trailing"
    assert on.globals.output_name == "out"


def test_full_line_comments_are_skipped():
    config = bd(
        """
        ## Energy doesn't actually get printed!
        outputEnergyPeriod 1000.0
        # another
        """
    )
    assert config.globals.output_energy_period == pytest.approx(1000.0)


# ===========================================================================
# the parser opens nothing but the .bd
# ===========================================================================


def test_parsing_never_opens_an_auxiliary_file(tmp_path):
    """Every filename stays an unresolved string.

    This is what lets a config with 140 MB of missing grids be a golden
    snapshot.
    """
    config = parse_string(
        textwrap.dedent(
            """
            particle B
            gridFile /nonexistent/null.dx
            rigidBody R
            potentialGrid k /nonexistent/grid.dx
            inputParticles /nonexistent/p.txt
            inputBonds /nonexistent/b.txt
            inputRBCoordinates /nonexistent/rb.txt
            tabulatedFile 0@0@/nonexistent/t.dat
            """
        ),
        source_path=str(tmp_path / "run.bd"),
    )
    assert config.particles[0].grid_files[0].filename == "/nonexistent/null.dx"
    assert config.input_particles == "/nonexistent/p.txt"
    assert config.topology_files[0].filename == "/nonexistent/b.txt"


def test_dry_run_resolution_opens_nothing():
    from marsmd.bd.apply import resolve_inputs

    config = bd(
        """
        particle B
        gridFile potentials/null.dx
        inputParticles /nonexistent/p.txt
        """
    )
    resolved = resolve_inputs(config, load=False)
    assert resolved.grid_paths == {"potentials/null.dx": "./potentials/null.dx"}
    assert resolved.particles == []


def test_resolution_keeps_unnormalized_grid_keys():
    from marsmd.bd.apply import resolve_inputs

    config = parse_string(
        "rigidBody R\npotentialGrid Pbead ../grids/grid-P.dx\n",
        source_path="sim/run.bd",
    )
    resolved = resolve_inputs(config, load=False)
    assert resolved.grid_paths == {"../grids/grid-P.dx": "sim/../grids/grid-P.dx"}


# ===========================================================================
# aux readers
# ===========================================================================


def test_read_particles_file(tmp_path):
    p = tmp_path / "particles.txt"
    p.write_text(
        "# comment\n"
        "ATOM 0 B 1.0 2.0 3.0\n"
        "ATOM 1 P 4.0 5.0 6.0 0.1 0.2 0.3\n"
        "\n"
        "short line\n"
    )
    records = aux.read_particles_file(str(p))
    assert len(records) == 2
    assert records[0].id == 0
    assert records[0].type_name == "B"
    assert records[0].position == (1.0, 2.0, 3.0)
    # Momentum columns are optional.
    assert records[0].momentum == (0.0, 0.0, 0.0)
    assert records[1].momentum == pytest.approx((0.1, 0.2, 0.3))


def test_read_restart_file_maps_type_ids_by_declaration_order(tmp_path):
    p = tmp_path / "restart.txt"
    p.write_text("0 1.0 2.0 3.0\n1 4.0 5.0 6.0\n9 7.0 8.0 9.0\n")
    records = aux.read_restart_file(str(p), ["B", "P"])
    assert [r.type_name for r in records] == ["B", "P"]
    # id is the line order among accepted lines.
    assert [r.id for r in records] == [0, 1]


def test_read_topology_file_dispatches_on_record_type(tmp_path):
    p = tmp_path / "topo.txt"
    p.write_text(
        """
        # a topology file
        BOND REPLACE 0 1 tab/bond.dat
        BOND ADD 1 2 tab/bond.dat
        BOND 2 3 Harmonic
        ANGLE 0 1 2 tab/angle.dat
        ANGLE 1 2 3 Harmonic
        DIHEDRAL 0 1 2 3 tab/dihedral.dat
        EXCLUDE 5 6
        RESTRAINT 4 10.0 1.0 2.0 3.0
        NONSENSE 1 2
        """.replace("        ", "")
    )
    topo = aux.read_topology_file(str(p))
    assert len(topo.bonds) == 3
    assert len(topo.angles) == 2
    assert len(topo.dihedrals) == 1
    assert len(topo.restraints) == 1


def test_bond_flag_is_optional():
    """Files written without a flag start straight at ind1."""
    from marsmd.bd.aux import _parse_bond

    flagged = _parse_bond(["REPLACE", "0", "1", "t.dat"])
    bare = _parse_bond(["0", "1", "t.dat"])
    assert flagged.flag == "REPLACE"
    assert bare.flag == "DEFAULT"
    assert (bare.ind1, bare.ind2) == (0, 1)


def test_replace_bond_implies_an_exclusion(tmp_path):
    """A REPLACE bond supersedes the pair's nonbonded term."""
    p = tmp_path / "t.txt"
    p.write_text("BOND REPLACE 0 1 t.dat\nBOND ADD 2 3 t.dat\n")
    topo = aux.read_topology_file(str(p))
    assert [(e.ind1, e.ind2) for e in topo.exclusions] == [(0, 1)]


def test_flagless_bond_also_implies_an_exclusion(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("BOND 0 1 t.dat\n")
    topo = aux.read_topology_file(str(p))
    assert [(e.ind1, e.ind2) for e in topo.exclusions] == [(0, 1)]


def test_analytical_names_resolve_to_an_index_tabulated_ones_do_not(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("BOND 0 1 Harmonic\nBOND 2 3 tab/bond.dat\nBOND 4 5 WLCSK\n")
    topo = aux.read_topology_file(str(p))
    assert (topo.bonds[0].form, topo.bonds[0].function_index) == ("Analytical", 0)
    # Only TablesRegistry can resolve a tabulated index.
    assert (topo.bonds[1].form, topo.bonds[1].function_index) == ("Tabulated", -1)
    assert (topo.bonds[2].form, topo.bonds[2].function_index) == ("Analytical", 4)


def test_invalid_bond_indices_are_dropped(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("BOND 0 0 t.dat\nBOND -1 2 t.dat\nBOND 1 2 t.dat\n")
    topo = aux.read_topology_file(str(p))
    assert [(b.ind1, b.ind2) for b in topo.bonds] == [(1, 2)]


def test_topology_files_accumulate_into_one_object(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("BOND 0 1 t.dat\n")
    b.write_text("ANGLE 0 1 2 t.dat\n")
    topo = aux.read_topology_file(str(a))
    aux.read_topology_file(str(b), topo)
    assert len(topo.bonds) == 1
    assert len(topo.angles) == 1
    assert len(topo) == 3  # bond + its implicit exclusion + angle


def test_read_rigid_body_coords(tmp_path):
    p = tmp_path / "rb.txt"
    p.write_text(
        "# comment\n"
        "1 2 3  1 0 0  0 1 0  0 0 1\n"
        "4 5 6  1 0 0  0 1 0  0 0 1  7 8 9  10 11 12\n"
    )
    coords = aux.read_rigid_body_coords(str(p))
    assert len(coords) == 2
    assert coords[0].position == (1.0, 2.0, 3.0)
    assert coords[0].orientation == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert coords[0].momentum is None
    assert coords[1].momentum == (7.0, 8.0, 9.0)
    assert coords[1].angular_momentum == (10.0, 11.0, 12.0)


def test_short_rigid_body_coord_line_is_an_error(tmp_path):
    p = tmp_path / "rb.txt"
    p.write_text("1 2 3\n")
    with pytest.raises(ValueError, match="at least 12"):
        aux.read_rigid_body_coords(str(p))


# ===========================================================================
# serialization
# ===========================================================================


def test_config_round_trips_to_json():
    config = bd(
        """
        temperature 295
        particle B
        num 500
        gridFile potentials/null.dx
        rigidBody SSB
        num 14
        potentialGrid Pbead ../grids/grid-P.dx
        """
    )
    payload = json.loads(config.to_json())
    assert payload["globals"]["temperature"] == 295.0
    assert payload["particles"][0]["name"] == "B"
    assert payload["rigid_bodies"][0]["potential_grids"][0]["key"] == "Pbead"


def test_json_is_stable_across_parses():
    text = "temperature 295\nparticle B\nnum 5\n"
    assert parse_string(text).to_json() == parse_string(text).to_json()
