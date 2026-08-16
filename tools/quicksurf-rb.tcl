##
## QuickSurf views of ARBD v2 rigid bodies.
##
## One ColorID per rigid-body type. In v2 every body lives in a single molecule,
## so `mol color Molecule` no longer separates them; the type is in the segname
## instead, and each segname gets its own ColorID, held for the session so a
## type keeps its colour across molecules and across template/run views.
##
## Two entry points:
##   rbTemplate <prefix>          one type's template, before any run
##   rbRun      <psf> ?<dcd>?     a finished run: every body, plus free particles
##
## usage:
##   vmd -e loading/quicksurf-rb.tcl -args template  rb_templates/1u8f.protein.att
##   vmd -e loading/quicksurf-rb.tcl -args templates rb_templates
##   vmd -e loading/quicksurf-rb.tcl -args run npc_2022/output/run.psf npc_2022/output/run.dcd
##
## interactively:
##   source loading/quicksurf-rb.tcl
##   rbTemplate rb_templates/2hiu.protein.att
##
## Conventions this relies on — see rb_templates/README.md, src/dev_notes.md:
##   segname ATT        a real, force-bearing attached particle
##   segname SYS        a free particle (run output only)
##   any other segname  a cosmetic atom, the segname being its rigid body's type
##   beta               the rigid body id, on attached and cosmetic atoms alike
##

set ::rbQuickSurf      {QuickSurf 1.0 0.5 1.9 1.0}
set ::rbQuickSurfCoarse {QuickSurf 1.0 0.5 3.0 1.0}
set ::rbMaterial       AOChalky
set ::rbAttachedRadius 2.0
set ::rbAttachedColor  4
set ::rbConfineColor   1

## ColorIDs handed out to types, in order. 8 (white) is omitted: it disappears
## against the white background this preset sets.
set ::rbColorIDs {0 1 3 4 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23}

## segname -> ColorID, persistent for the session.
if {![info exists ::rbSegColor]} { set ::rbSegColor [dict create] }

## Every segname that is not one of the reserved tags: one per rigid-body type.
proc rbTypeSegnames {molid} {
    set sel [atomselect $molid "not segname ATT SYS"]
    set names [lsort -unique [$sel get segname]]
    $sel delete
    return $names
}

proc rbCountSel {molid text} {
    set sel [atomselect $molid $text]
    set n [$sel num]
    $sel delete
    return $n
}

## ColorID for a type's segname, assigned on first sight and kept thereafter.
proc rbColorFor {seg} {
    if {![dict exists $::rbSegColor $seg]} {
        set n [dict size $::rbSegColor]
        dict set ::rbSegColor $seg \
            [lindex $::rbColorIDs [expr {$n % [llength $::rbColorIDs]}]]
    }
    return [dict get $::rbSegColor $seg]
}

proc rbAddRep {molid rep sel color {material ""}} {
    if {$material eq ""} { set material $::rbMaterial }
    mol representation $rep
    mol selection $sel
    mol color $color
    mol material $material
    mol addrep $molid
}

##
## One rigid-body template: PSF + PDB sharing a prefix.
##
proc rbTemplate {prefix args} {
    set quicksurf $::rbQuickSurf
    foreach {k v} $args { if {$k eq "-quicksurf"} { set quicksurf $v } }

    set molid [mol new $prefix.psf type psf waitfor all]
    mol addfile $prefix.pdb type pdb waitfor all $molid
    mol rename $molid [file tail $prefix]
    mol delrep 0 $molid

    set segs [rbTypeSegnames $molid]
    foreach seg $segs {
        rbAddRep $molid $quicksurf "segname $seg" "ColorID [rbColorFor $seg]"
    }
    rbAddRep $molid "VDW $::rbAttachedRadius 20" {segname ATT} \
        "ColorID $::rbAttachedColor"
    rbAddRep $molid "VDW [expr {$::rbAttachedRadius * 1.5}] 20" \
        {segname ATT and resname CONFINE} "ColorID $::rbConfineColor"

    puts "[file tail $prefix]: [rbCountSel $molid all] atoms,\
[rbCountSel $molid {segname ATT}] attached"
    foreach seg $segs { puts "  segname $seg -> ColorID [rbColorFor $seg]" }
    return $molid
}

## Every template in a directory, one molecule each, sharing the colour map.
proc rbTemplateDir {{dir rb_templates}} {
    set ids {}
    foreach psf [lsort [glob -nocomplain $dir/*.att.psf]] {
        lappend ids [rbTemplate [file rootname $psf]]
    }
    rbDisplayPreset
    if {[llength $ids] > 0} { mol top [lindex $ids 0] }
    display resetview
    return $ids
}

##
## A finished run: the PSF ARBD wrote, plus its DCD.
##
proc rbRun {psf {dcd ""} args} {
    set quicksurf ""
    foreach {k v} $args { if {$k eq "-quicksurf"} { set quicksurf $v } }

    set molid [mol new $psf type psf waitfor all]
    if {$dcd ne ""} { mol addfile $dcd type dcd waitfor all $molid }
    mol rename $molid [file tail $psf]
    mol delrep 0 $molid

    set segs [rbTypeSegnames $molid]
    set nrb [rbCountSel $molid {not segname ATT SYS}]
    if {$quicksurf eq ""} {
        set quicksurf [expr {$nrb > 1000000 ? $::rbQuickSurfCoarse : $::rbQuickSurf}]
    }

    ## One rep per type, so a type can be hidden on its own. Within a type the
    ## instances are spatially disjoint, so the single rep still draws each
    ## body as its own surface.
    foreach seg $segs {
        rbAddRep $molid $quicksurf "segname $seg" "ColorID [rbColorFor $seg]"
    }
    rbAddRep $molid "VDW $::rbAttachedRadius 12" {segname ATT} \
        "ColorID $::rbAttachedColor"
    if {[rbCountSel $molid {segname SYS}] > 0} {
        rbAddRep $molid {Points 4} {segname SYS} Name
    }

    puts "[file tail $psf]: [rbCountSel $molid all] atoms\
([rbCountSel $molid {segname SYS}] free,\
 [rbCountSel $molid {segname ATT}] attached,\
 $nrb cosmetic)"
    foreach seg $segs { puts "  segname $seg -> ColorID [rbColorFor $seg]" }
    puts "  quicksurf:  $quicksurf"
    puts "  one type:   segname [lindex $segs 0]"
    puts "  one body:   beta 0"
    puts "  its beads:  segname ATT and beta 0"

    rbDisplayPreset
    mol top $molid
    display resetview
    return $molid
}

## Switch a loaded molecule from one-colour-per-type to one-colour-per-body.
proc rbColorByBody {molid} {
    for {set r 0} {$r < [molinfo $molid get numreps]} {incr r} {
        mol modcolor $r $molid Beta
    }
}

## ...and back.
proc rbColorByType {molid} {
    set r 0
    foreach seg [rbTypeSegnames $molid] {
        mol modcolor $r $molid "ColorID [rbColorFor $seg]"
        incr r
    }
}

## ── Display preset: publication ─────────────────────────────────
proc rbDisplayPreset {} {
    color Display Background white
    display projection   Perspective
    display depthcue     on
    display cuedensity   0.22
    display backgroundgradient off
    display shadows      on
    display ambientocclusion on
    display aoambient    1.6
    display aodirect     0.0
    display dof_focaldist 2.0
    display dof_fnumber  1200
    catch {display rendermode   GLSL}
    catch {display resize 2160 2160}
    axes location Off
    mol material $::rbMaterial
}

if {[info exists argv] && [llength $argv] > 0} {
    switch -- [lindex $argv 0] {
        template  {
            set m [rbTemplate [lindex $argv 1]]
            rbDisplayPreset
            mol top $m
            display resetview
        }
        templates {
            if {[llength $argv] > 1} { rbTemplateDir [lindex $argv 1] } else { rbTemplateDir }
        }
        run       { rbRun [lindex $argv 1] [lindex $argv 2] }
        default   { puts "usage: -args template <prefix> | templates ?<dir>? | run <psf> ?<dcd>?" }
    }
}
