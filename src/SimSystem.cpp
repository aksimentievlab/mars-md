#include "SimSystem.h"

void CellDecomposer::decompose(SimSystem& sys, ResourceCollection& resources) {
    BoundaryConditions& bcs = sys.boundary_conditions;
    const Length& cutoff = sys.cutoff;

    // Initialize workers; TODO move this out of decompose
    std::vector<Proxy<Worker>> workers;
    for (auto& r: resources.resources) {
	Worker w{sys.cutoff, sys.patches};
	auto w_p = send(r, w);
	workers.push_back( w_p );
    }
    
    Vector3 min = sys.get_min();
    Vector3 max = sys.get_max();
    Vector3 dr = max-min;

    // For starters, distribute patches uniformly among available resources
    Vector3 n_p_v = (dr / cutoff).element_floor();
    size_t n_r = resources.resources.size();

    size_t num_particles = 0;
    for (auto& p: sys.patches) num_particles += p.metadata->num;
   
    std::vector<Proxy<Patch>> new_patches;
    // std::vector<Patch> new_patches;
    size_t n_p = static_cast<size_t>(round(n_p_v.x*n_p_v.y*n_p_v.z));
    for (size_t idx = 0; idx < n_p; ++idx) {
	Vector3_t<size_t> ijk = index_to_ijk( idx, n_p_v.x, n_p_v.y, n_p_v.z );
	// size_t cap = 2*num_particles / n_r;
	// auto p2 = Patch(cap);
	Patch p2 = Patch();	// don't allocate array locally
	// TODO: generalize to non-orthogonal basis
	Vector3 pmin = min + dr.element_mult(ijk);
	Vector3 pmax = pmin + dr;

	p2.lower_bound = pmin;
	p2.upper_bound = pmax;

	size_t r_idx = idx / ceil(n_p/n_r);
	
	//auto p2_p = send(resources.resources[r_idx], p2);
	Proxy<Patch> p2_p = send(resources.resources[r_idx], p2);

	for (auto& p: sys.patches) {
	    auto filter_lambda = [pmin, pmax] (size_t i, Patch::Data d)->bool {
		return d.get_pos(i).x >= pmin.x && d.get_pos(i).x < pmax.x &&
		    d.get_pos(i).y >= pmin.y && d.get_pos(i).y < pmax.y &&
		    d.get_pos(i).z >= pmin.z && d.get_pos(i).z < pmax.z; };
	    auto filter = std::function<bool(size_t, Patch::Data)>(filter_lambda);
	    using _filter_t = decltype(filter);
	    // p.callSync<void,Proxy<Patch>,_filter_t>( &Patch::send_particles_filtered<_filter_t>, p2_p, filter );
	    // p.callSync<size_t, Proxy<Patch>&, _filter_t>(static_cast<size_t (Patch::*)(Proxy<Patch>&, _filter_t)>(&Patch::send_particles_filtered<_filter_t>), p2_p, filter);
	    // p.callSync<size_t, Proxy<Patch>&>( &Patch::send_particles_filtered, p2_p, filter);
	    p.callSync( &Patch::send_particles_filtered, p2_p, filter);
	    
	    num_particles += p.metadata->num;
	}
	
	new_patches.push_back( p2_p );
    }
    
    // size_t n_p = static_cast<size_t>(round(n_p_v[0]*n_p_v[1]*n_p_v[2]));
    // for (size_t i = 0; i < n_p; ++i) {
    // 	auto w_p = workers[i / floor(n_p/n_r)];
    // 	w_p.create_patch();
    // 	// other stuff
    // }
	
    
    
    // Count particles in each new_patch
    //   (ultimately this must be distributed, but via what mechanisms? Decomposer is only on controlling thread in current implementation, but perhaps each thread should have its own copy)
    // Then add particles from old patches to new one
    
    sys.patches = new_patches;
}
