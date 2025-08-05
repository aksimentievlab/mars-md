#include "molfile_plugin.h"
#include "libmolfile_plugin.h"
#include "useful.h"

static molfile_plugin_t *dcdplugin;
static int register_cb(void *v, vmdplugin_t *p) {
	dcdplugin = (molfile_plugin_t *)p;
	return 0;
}
static int numatoms;
static void *filehandle;
static float *coords;
static Vector3 *vcoords;


class DcdReader {
public:
    DcdReader(const char* filename) {
	molfile_dcdplugin_init();
	molfile_dcdplugin_register(NULL, register_cb);

	filehandle = dcdplugin->open_file_read(filename, "dcd", &numatoms);

	if (!filehandle) {
	    printf("DcdReader: Error opening file %s\n", filename);
	    exit(1);
	}

	/* Atom check
	if (numatoms != ()->pdb->num_atoms()) {
	    Tcl_AppendResult(interp, "Coordinate file ", argv[3],
			     "\ncontains the wrong number of atoms.", NULL);
	    return TCL_ERROR;
	}
	*/ 
	coords = new float[3*numatoms];
	vcoords = new Vector3[numatoms];
	printf("Coordinate file %s opened for reading.\n", filename);
    }
  ~DcdReader() {
      if (!filehandle) {
	  printf("DcdReader: No file opened for reading!\n");
	  exit(1);
      }
      printf("Closing coordinate file.\n");
      dcdplugin->close_file_read(filehandle);
      filehandle = NULL;
      delete [] coords;
      delete [] vcoords;
  }
  Vector3* read_step() {
	if (filehandle == NULL) {
	    printf("DcdReader: Error, no file open for reading\n");
	    exit(1);
	}
	molfile_timestep_t ts;
	ts.coords = coords;
	int rc = dcdplugin->read_next_timestep(filehandle, numatoms, &ts);
	if (rc) {  // EOF
	    return NULL;
	}
	// printf("Reading timestep from file.\n");
	
	/*
	Lattice lattice;
	if (get_lattice_from_ts(&lattice, &ts)) {
	    iout << iINFO << "Updating unit cell from timestep.\n" << endi;
	    if ( lattice.a_p() && ! script->state->lattice.a_p() ||
		 lattice.b_p() && ! script->state->lattice.b_p() ||
		 lattice.c_p() && ! script->state->lattice.c_p() ) {
		iout << iWARN << "Cell basis vectors should be specified before reading trajectory.\n" << endi;
	    }
	    // update Controller's lattice, but don't change the origin!
	    Vector a(0.);  if ( script->state->lattice.a_p() ) a = lattice.a();
	    Vector b(0.);  if ( script->state->lattice.b_p() ) b = lattice.b();
	    Vector c(0.);  if ( script->state->lattice.c_p() ) c = lattice.c();
	    script->state->lattice.set(a,b,c);
	    SetLatticeMsg *msg = new SetLatticeMsg;
	    msg->lattice = script->state->lattice;
	    (CProxy_PatchMgr(CkpvAccess(BOCclass_group).patchMgr)).setLattice(msg);
	    script->barrier();
	}
	*/
	for (int i=0; i<numatoms; i++) {
	    vcoords[i].x = coords[3*i+0];
	    vcoords[i].y = coords[3*i+1];
	    vcoords[i].z = coords[3*i+2];
	}
	/*
	Node::Object()->pdb->set_all_positions(vcoords);
	script->reinitAtoms();
	Tcl_SetObjResult(interp, Tcl_NewIntObj(0));
	*/
	return vcoords;
  }
  bool skip_step() {
	if (filehandle == NULL) {
	    printf("DcdReader: Error, no file open for reading\n");
	    exit(1);
	}
	int rc = dcdplugin->read_next_timestep(filehandle, numatoms, NULL);
	if (rc) {  // EOF
	    return false;
	}
	// printf("Skipping step in file.\n");
	return true;
  }
};
