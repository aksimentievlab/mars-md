#!/usr/bin/env python3
"""
Documentation generation script for C++ ARBD code.

This script:
1. Sets up a Doxygen configuration
2. Organizes source files into logical groups
3. Generates reference documentation
4. Creates a simple tutorial from test examples
5. Builds a searchable documentation website
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import argparse
import re
import glob

# Path configuration - update these to match your local setup
ARBD_CPP_PATH = Path(".")  # Current directory, adjust as needed
DOCS_DIR = Path("docs")
SOURCE_DIR = ARBD_CPP_PATH / "src"
TEST_DIR = ARBD_CPP_PATH / "tests"
OUTPUT_DIR = DOCS_DIR / "html"

# Define the structure of code modules
CODE_STRUCTURE = {
    "Core": [
        "GrandBrownTown.h",
        "GrandBrownTown.cu",
        "GrandBrownTown.cuh",
        "Configuration.h",
        "Configuration.cpp",
        "arbd.cpp",
    ],
    "Particles": [
        "BrownianParticle.h",
        "BrownianParticle.cu",
        "BrownianParticleType.h",
        "BrownianParticleType.cpp",
        "BrownParticlesKernel.h"
    ],
    "RigidBodies": [
        "RigidBody.h",
        "RigidBody.cu",
        "RigidBodyType.h",
        "RigidBodyType.cu",
        "RigidBodyController.h",
        "RigidBodyController.cu",
        "RigidBodyGrid.h",
        "RigidBodyGrid.cu",
    ],
    "Potentials": [
        "TabulatedPotential.h",
        "TabulatedPotential.cu",
        "TabulatedMethods.cuh",
        "TabulatedAngle.h",
        "TabulatedAngle.cu",
        "TabulatedDihedral.h",
        "TabulatedDihedral.cu",
    ],
    "Interactions": [
        "Angle.h",
        "Angle.cu",
        "Dihedral.h",
        "Dihedral.cu",
        "Exclude.h",
        "Exclude.cu",
        "JamesBond.h",
        "JamesBond.cu",
        "ProductPotential.h",
    ],
    "Grids": [
        "BaseGrid.h",
        "BaseGrid.cu",
        "ComputeGridGrid.cu",
        "ComputeGridGrid.cuh",
        "OverlordGrid.h",
    ],
    "Forces": [
        "ComputeForce.h",
        "ComputeForce.cu",
        "ComputeForce.cuh",
        "ComputeJustForce.h",
        "FlowForce.h",
        "FlowForce.cpp",
    ],
    "Utils": [
        "CudaUtil.cu",
        "CudaUtil.cuh",
        "WKFUtils.h",
        "WKFUtils.cpp",
        "useful.h",
        "useful.cu",
        "Random.h",
        "RandomCPU.h",
        "RandomCUDA.h",
        "RandomCUDA.cu",
        "Debug.h",
    ],
    "IO": [
        "Reader.h",
        "DcdWriter.h",
        "TrajectoryWriter.h",
        "vmdsock.h",
        "vmdsock.cpp",
        "imd.h",
        "imd.cpp",
    ],
    "GPU": [
        "GPUController.h",
        "GPUManager.h",
        "GPUManager.cpp",
        "CellDecomposition.h",
        "CellDecomposition.cu",
    ],
    "Other": [
        "Restraint.h",
        "Reservoir.h",
        "Reservoir.cu",
        "Scatter.h",
        "Scatter.cpp",
        "SignalManager.h",
        "SignalManager.cpp",
    ]
}

# Define example tests to document as tutorials
TUTORIAL_TESTS = [
    {"name": "Running a Simple Argon Simulation", "dir": "argon-small"},
    {"name": "Simulating with Bonded Interactions", "dir": "bond"},
    {"name": "Langevin Dynamics Simulation", "dir": "langevin"},
    {"name": "Non-bonded Pair Interactions", "dir": "nb-pair"}
]

def setup_directories():
    """Create necessary directories for documentation."""
    print("\nSetting up documentation directories...")
    
    # Create main documentation directory
    DOCS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories for different content
    tutorial_dir = DOCS_DIR / "tutorials"
    tutorial_dir.mkdir(exist_ok=True)
    
    api_dir = DOCS_DIR / "api"
    api_dir.mkdir(exist_ok=True)
    
    # Directory for test examples
    example_dir = DOCS_DIR / "examples"
    example_dir.mkdir(exist_ok=True)
    
    print(f"Created directories at {DOCS_DIR}")
    return tutorial_dir, api_dir, example_dir

def check_doxygen():
    """Check if Doxygen is installed."""
    try:
        result = subprocess.run(['doxygen', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"Found Doxygen: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("Doxygen not found! Please install Doxygen to generate C++ documentation.")
        print("Installation instructions: https://www.doxygen.nl/download.html")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error checking Doxygen: {e}")
        return False

def create_doxygen_config():
    """Create a Doxygen configuration file."""
    print("\nCreating Doxygen configuration...")
    
    config_path = DOCS_DIR / "Doxyfile"
    
    # Start with default configuration
    try:
        subprocess.run(['doxygen', '-g', str(config_path)], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating Doxygen configuration: {e}")
        return None
    
    # Read the generated config
    with open(config_path, 'r') as f:
        config = f.read()
    
    # Update configuration
    config = re.sub(r'PROJECT_NAME\s*=.*', f'PROJECT_NAME = "ARBD C++ Library"', config)
    config = re.sub(r'PROJECT_BRIEF\s*=.*', f'PROJECT_BRIEF = "Atomic Resolution Brownian Dynamics"', config)
    config = re.sub(r'OUTPUT_DIRECTORY\s*=.*', f'OUTPUT_DIRECTORY = "{DOCS_DIR}"', config)
    config = re.sub(r'INPUT\s*=.*', f'INPUT = "{SOURCE_DIR}"', config)
    config = re.sub(r'RECURSIVE\s*=.*', 'RECURSIVE = YES', config)
    config = re.sub(r'EXTRACT_ALL\s*=.*', 'EXTRACT_ALL = YES', config)
    config = re.sub(r'EXTRACT_PRIVATE\s*=.*', 'EXTRACT_PRIVATE = YES', config)
    config = re.sub(r'EXTRACT_STATIC\s*=.*', 'EXTRACT_STATIC = YES', config)
    config = re.sub(r'HAVE_DOT\s*=.*', 'HAVE_DOT = YES', config)
    config = re.sub(r'CALL_GRAPH\s*=.*', 'CALL_GRAPH = YES', config)
    config = re.sub(r'CALLER_GRAPH\s*=.*', 'CALLER_GRAPH = YES', config)
    config = re.sub(r'UML_LOOK\s*=.*', 'UML_LOOK = YES', config)
    config = re.sub(r'SOURCE_BROWSER\s*=.*', 'SOURCE_BROWSER = YES', config)
    config = re.sub(r'ALPHABETICAL_INDEX\s*=.*', 'ALPHABETICAL_INDEX = YES', config)
    config = re.sub(r'GENERATE_TREEVIEW\s*=.*', 'GENERATE_TREEVIEW = YES', config)
    config = re.sub(r'SEARCHENGINE\s*=.*', 'SEARCHENGINE = YES', config)
    config = re.sub(r'GENERATE_LATEX\s*=.*', 'GENERATE_LATEX = NO', config)
    config = re.sub(r'USE_MDFILE_AS_MAINPAGE\s*=.*', 'USE_MDFILE_AS_MAINPAGE = README.md', config)
    config = re.sub(r'HTML_EXTRA_STYLESHEET\s*=.*', f'HTML_EXTRA_STYLESHEET = {DOCS_DIR / "doxygen-awesome.css"}', config)
    
    # Enable parsing of CUDA files
    config = re.sub(r'FILE_PATTERNS\s*=.*', 'FILE_PATTERNS = *.c *.cc *.cxx *.cpp *.h *.hh *.hpp *.cu *.cuh', config)
    
    # Add grouping for modules
    group_definitions = "\n"
    for group, files in CODE_STRUCTURE.items():
        group_definitions += f'# {group} group\n'
        group_definitions += f'GROUP_{group.upper().replace(" ", "_")} = \\\n'
        for file in files:
            group_definitions += f'    {SOURCE_DIR / file} \\\n'
        group_definitions += "\n"
    
    # Add group definitions
    config += "\n# Group definitions\n" + group_definitions
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f"Created Doxygen configuration at {config_path}")
    return config_path

def create_doxygen_theme():
    """Download and set up the Doxygen Awesome theme."""
    theme_url = "https://github.com/jothepro/doxygen-awesome-css/raw/main/doxygen-awesome.css"
    theme_path = DOCS_DIR / "doxygen-awesome.css"
    
    try:
        subprocess.run(['wget', '-O', str(theme_path), theme_url],) 
        print(f"Downloaded Doxygen theme to {theme_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading theme: {e}")
        print("Will use default Doxygen theme instead.")
    except FileNotFoundError:
        print("curl command not found. Using default Doxygen theme.")
        try:
            # Try wget as an alternative
            subprocess.run(['curl', '-O', str(theme_path), theme_url], 
                          check=True, capture_output=True)
            print(f"Downloaded Doxygen theme to {theme_path}")
        except:
            print("wget command also not found. Using default Doxygen theme.")

def create_mainpage():
    """Create the main page markdown for Doxygen."""
    print("Creating main documentation page...")
    
    content = """# ARBD C++ Library Documentation

Welcome to the documentation for the ARBD (Acceleration of Rigid Body Dynamics) C++ library!

## Overview

The ARBD library provides high-performance molecular dynamics simulations using CUDA acceleration. It is designed to efficiently handle both particle-based and rigid-body simulations for a variety of molecular systems.

## Key Features

- CUDA-accelerated molecular dynamics simulations
- Support for particle-based and rigid body simulations
- Tabulated potentials for flexible interaction modeling
- Grid-based acceleration for large systems
- Various integration methods including Brownian dynamics and Langevin dynamics

## Getting Started

To get started with ARBD, check out the tutorials section which walks through example simulations.

## Module Organization

The ARBD codebase is organized into the following modules:

- **Core**: Main simulation driver classes
- **Particles**: Brownian particle implementations
- **RigidBodies**: Rigid body simulation components
- **Potentials**: Tabulated and other potential implementations
- **Interactions**: Various interaction types (bonds, angles, dihedrals)
- **Grids**: Grid-based acceleration structures
- **Forces**: Force computation routines
- **Utils**: Utility classes and functions
- **IO**: Input/output functionality
- **GPU**: GPU management and cell decomposition

## Examples

The `tests` directory contains several example simulations that demonstrate how to use the library:

- Simple Argon simulation
- Systems with bonded interactions
- Langevin dynamics
- Non-bonded pair interactions

These examples are documented in the Tutorials section.
"""
    
    mainpage_path = DOCS_DIR / "mainpage.md"
    with open(mainpage_path, 'w') as f:
        f.write(content)
    
    print(f"Created main page at {mainpage_path}")
    return mainpage_path

def copy_test_examples(example_dir):
    """Copy and document test examples."""
    print("\nProcessing test examples...")
    
    for tutorial in TUTORIAL_TESTS:
        name = tutorial["name"]
        test_dir = tutorial["dir"]
        
        # Create directory for this example
        target_dir = example_dir / test_dir
        target_dir.mkdir(exist_ok=True)
        
        # Copy relevant files
        source_test_dir = TEST_DIR / test_dir
        if not source_test_dir.exists():
            print(f"Warning: Test directory {source_test_dir} not found. Skipping.")
            continue
        
        # Copy all non-binary files
        for file_path in source_test_dir.glob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                # Skip large binary files
                if file_path.suffix not in ['.dcd', '.psf', '.coor', '.vel']:
                    shutil.copy2(file_path, target_dir / file_path.name)
        
        # Copy analysis directory if it exists
        analysis_dir = source_test_dir / "analysis"
        if analysis_dir.exists():
            target_analysis_dir = target_dir / "analysis"
            target_analysis_dir.mkdir(exist_ok=True)
            
            for file_path in analysis_dir.glob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    shutil.copy2(file_path, target_analysis_dir / file_path.name)
        
        print(f"Copied example files for: {name}")
    
    # Create index file for examples
    create_examples_index(example_dir)

def create_examples_index(example_dir):
    """Create an index file for examples."""
    content = """# ARBD Examples

This section contains example simulations that demonstrate how to use the ARBD library.

"""
    
    for tutorial in TUTORIAL_TESTS:
        name = tutorial["name"]
        test_dir = tutorial["dir"]
        
        content += f"## {name}\n\n"
        
        # Add description based on BD file content
        bd_file = example_dir / test_dir / "BrownDyn.bd"
        if bd_file.exists():
            try:
                with open(bd_file, 'r') as f:
                    bd_content = f.read()
                    # Extract any comments at the top
                    comments = []
                    for line in bd_content.split('\n'):
                        if line.strip().startswith('#'):
                            comments.append(line.strip('# '))
                        else:
                            break
                    
                    if comments:
                        content += '\n'.join(comments) + "\n\n"
            except:
                pass
        
        content += f"Files for this example can be found in the `tests/{test_dir}` directory.\n\n"
        
        # List key files
        content += "**Key files:**\n\n"
        
        for file_name in ["BrownDyn.bd", "run.sh"]:
            file_path = example_dir / test_dir / file_name
            if file_path.exists():
                content += f"- [{file_name}](examples/{test_dir}/{file_name})\n"
        
        # Analysis files if they exist
        analysis_dir = example_dir / test_dir / "analysis"
        if analysis_dir.exists():
            content += "\n**Analysis scripts:**\n\n"
            for file_path in analysis_dir.glob("*.tcl"):
                content += f"- [analysis/{file_path.name}](examples/{test_dir}/analysis/{file_path.name})\n"
        
        content += "\n"
    
    index_path = example_dir / "index.md"
    with open(index_path, 'w') as f:
        f.write(content)
    
    print(f"Created examples index at {index_path}")

def create_tutorials():
    """Create tutorial markdown files."""
    print("\nCreating tutorials...")
    
    tutorials_dir = DOCS_DIR / "tutorials"
    tutorials_dir.mkdir(exist_ok=True)
    
    # Create the getting started tutorial
    getting_started = """# Getting Started with ARBD

This tutorial will guide you through the basics of using the ARBD library for molecular dynamics simulations.

## Prerequisites

Before you begin, make sure you have:

1. Compiled the ARBD library successfully
2. CUDA toolkit installed (recommended version: 10.0 or newer)
3. Basic understanding of molecular dynamics concepts

## Basic Simulation Structure

All ARBD simulations follow this general structure:

1. Define a simulation configuration (`.bd` file)
2. Prepare input coordinates and parameters
3. Run the simulation using the `arbd` executable
4. Analyze the results with post-processing scripts

## Example: Simple Argon Simulation

Let's walk through a simple Argon simulation as an example.

### 1. Configuration File

The simulation is defined in a `.bd` file. Here's an example from the argon-small test:

```
seed 20101992
timestep 0.01
steps 10000
interparticleForce 1
fullLongRange 0
temperature 120.0
ParticleDynamicType Brownian

outputPeriod 1000
outputEnergyPeriod 1000

decompPeriod 40
cutoff 2.5
pairlistDistance 0.5

systemSize 5 5 5

particle argon
num 100
diffusion 1
gridFile null.dx
```

This configuration:
- Sets a random seed
- Defines timestep and number of steps
- Specifies Brownian dynamics
- Sets output frequency
- Defines system size and parameters
- Creates 100 argon particles

### 2. Running the Simulation

Use the `arbd` executable to run the simulation:

```bash
./arbd BrownDyn.bd output/arbd
```

This will run the simulation and output files to the `output/arbd` directory.

### 3. Analyzing Results

The output includes trajectory files that can be analyzed with VMD or custom scripts.

## Next Steps

Check out the other examples in the Examples section to learn about:

- Bonded interactions
- Langevin dynamics
- Non-bonded pair interactions

Each example includes a complete configuration file and analysis scripts.
"""
    
    with open(tutorials_dir / "getting_started.md", 'w') as f:
        f.write(getting_started)
    
    # Create a configuration tutorial
    config_tutorial = """# ARBD Configuration Reference

This tutorial explains the configuration parameters used in ARBD `.bd` files.

## Basic Parameters

- `seed <number>`: Random number generator seed
- `timestep <value>`: Simulation timestep
- `steps <number>`: Total number of simulation steps
- `temperature <value>`: Simulation temperature (K)
- `ParticleDynamicType <type>`: Integration method (Brownian, Langevin, etc.)

## Output Settings

- `outputPeriod <number>`: How frequently to save trajectory frames
- `outputEnergyPeriod <number>`: How frequently to save energy data
- `outputFormat <format>`: Output file format (default: dcd)

## System Settings

- `systemSize <x> <y> <z>`: Simulation box dimensions
- `cutoff <value>`: Cutoff distance for non-bonded interactions
- `pairlistDistance <value>`: Extra distance for neighborlist construction
- `decompPeriod <number>`: How frequently to update domain decomposition

## Particle Definitions

Each particle type is defined with:

```
particle <name>
num <count>
diffusion <value>  # For Brownian dynamics
mass <value>       # For Langevin dynamics
transDamping <x> <y> <z>  # For Langevin dynamics
gridFile <filename>  # External potential grid
```

## Interactions

- `tabulatedPotential <0/1>`: Enable tabulated potentials
- `tabulatedFile <i>@<j>@<filename>`: Define interaction between particle types i and j

### Bonded Interactions

- `inputBonds <filename>`: File with bond definitions
- `inputAngles <filename>`: File with angle definitions
- `inputDihedrals <filename>`: File with dihedral definitions
- `inputExcludes <filename>`: File with exclusion definitions

## Advanced Settings

- `fullLongRange <0/1>`: Enable full long-range interactions
- `interparticleForce <0/1>`: Enable inter-particle forces
- `numberFluct <0/1>`: Enable number fluctuations

For more details, refer to the specific examples in the Examples section.
"""
    
    with open(tutorials_dir / "configuration.md", 'w') as f:
        f.write(config_tutorial)
    
    # Create a file format tutorial
    file_format_tutorial = """# ARBD File Formats

This tutorial explains the file formats used by ARBD.

## Coordinate Files

ARBD uses simple text-based coordinate files:

```
ATOM 0 argon 1.0 2.0 3.0
ATOM 1 argon 4.0 5.0 6.0
...
```

Each line specifies:
- ATOM keyword
- Atom index
- Atom type
- X, Y, Z coordinates

## Trajectory Files

ARBD outputs trajectories in DCD format, compatible with VMD and other analysis tools.

## Tabulated Potential Files

Tabulated potentials use two-column format:

```
# r    U(r)
0.0    100.0
0.1    90.0
...
```

The first column is the distance, the second is the potential value.

## Bond Files

Bond files define bonded interactions:

```
BOND ADD 0 1 bond-potential.dat
```

This defines a bond between atoms 0 and 1 using the potential in the specified file.

## APBS Grid Files

ARBD can use 3D potential grids in the APBS DX format for external potentials.

## Other File Formats

For more specialized formats, refer to the example files in the test directories.
"""
    
    with open(tutorials_dir / "file_formats.md", 'w') as f:
        f.write(file_format_tutorial)
    
    # Create a tutorial index
    tutorials_index = """# ARBD Tutorials

Welcome to the ARBD tutorials! These guides will help you understand how to use the ARBD library for molecular dynamics simulations.

## Basic Tutorials

- [Getting Started](tutorials/getting_started.md): Introduction to ARBD and first simulation
- [Configuration Reference](tutorials/configuration.md): Detailed guide to ARBD configuration parameters
- [File Formats](tutorials/file_formats.md): Reference for ARBD file formats

## Example Simulations

Please see the [Examples](examples/index.md) section for detailed walkthroughs of specific simulation types.

## Advanced Topics

For detailed information about the code structure and APIs, refer to the API documentation.
"""
    
    with open(tutorials_dir / "index.md", 'w') as f:
        f.write(tutorials_index)
    
    print(f"Created tutorials at {tutorials_dir}")

def run_doxygen(config_path):
    """Run Doxygen to generate documentation."""
    print("\nRunning Doxygen to generate documentation...")
    
    try:
        result = subprocess.run(['doxygen', str(config_path)],
                              check=True, capture_output=True, text=True)
        print("Doxygen documentation generation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running Doxygen: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def generate_module_headers():
    """Generate header files for documentation modules."""
    print("\nGenerating module headers...")
    
    module_dir = DOCS_DIR / "modules"
    module_dir.mkdir(exist_ok=True)
    
    for group, files in CODE_STRUCTURE.items():
        filename = module_dir / f"{group.lower()}_module.h"
        
        content = f"""/**
 * @defgroup {group.lower()} {group}
 * @brief {group} components of the ARBD library
 *
 * This module contains components related to {group.lower()} functionality.
 */

"""
        for file in files:
            file_path = SOURCE_DIR / file
            if file_path.exists():
                basename = Path(file).stem
                content += f"// @include {file}\n"
        
        with open(filename, 'w') as f:
            f.write(content)
    
    print(f"Generated module headers in {module_dir}")

def create_overall_index():
    """Create the overall documentation index."""
    content = """# ARBD Documentation

Welcome to the ARBD (Acceleration of Rigid Body Dynamics) documentation!

## Contents

- [Tutorials](tutorials/index.md): Guides for getting started with ARBD
- [Examples](examples/index.md): Example simulations
- [API Reference](html/index.html): Detailed code documentation

## Getting Started

If you're new to ARBD, start with the [Getting Started](tutorials/getting_started.md) tutorial.
"""
    
    index_path = DOCS_DIR / "index.md"
    with open(index_path, 'w') as f:
        f.write(content)
    
    print(f"Created main index at {index_path}")

def main():
    """Main function to generate C++ ARBD documentation."""
    parser = argparse.ArgumentParser(description="Generate ARBD C++ documentation")
    parser.add_argument('--output-dir', type=str, help='Output directory for documentation')
    parser.add_argument('--source-dir', type=str, help='Source code directory')
    parser.add_argument('--clean', action='store_true', help='Clean existing documentation directory')
    args = parser.parse_args()
    
    # Update paths if provided
    global DOCS_DIR, SOURCE_DIR, TEST_DIR
    
    if args.output_dir:
        DOCS_DIR = Path(args.output_dir)
    
    if args.source_dir:
        SOURCE_DIR = Path(args.source_dir)
        # Adjust TEST_DIR too, assuming standard structure
        TEST_DIR = SOURCE_DIR.parent / "tests"
    
    # Check if Doxygen is installed
    if not check_doxygen():
        print("Doxygen is required to generate C++ documentation.")
        sys.exit(1)
    
    # Clean documentation directory if requested
    if args.clean and DOCS_DIR.exists():
        print(f"Cleaning documentation directory: {DOCS_DIR}")
        shutil.rmtree(DOCS_DIR)
    
    # Setup directories
    tutorial_dir, api_dir, example_dir = setup_directories()
    
    # Create Doxygen theme
    create_doxygen_theme()
    
    # Create main documentation page
    create_mainpage()
    
    # Generate module headers
    generate_module_headers()
    
    # Create Doxygen configuration
    config_path = create_doxygen_config()
    
    # Create tutorials
    create_tutorials()
    
    # Copy test examples
    copy_test_examples(example_dir)
    
    # Create overall index
    create_overall_index()
    
    # Run Doxygen
    if config_path:
        if run_doxygen(config_path):
            print(f"\nDocumentation successfully generated at {OUTPUT_DIR}")
            print(f"Open {OUTPUT_DIR}/index.html in your browser to view it.")
        else:
            print("\nError generating documentation.")
            return 1
    
    print("\nDocumentation process completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
