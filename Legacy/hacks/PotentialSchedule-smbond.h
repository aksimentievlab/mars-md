#pragma once
#include "useful.h"
#include <cassert>
#include <vector>

struct PotentialSchedule {
    struct ValueAtStep {
	size_t step;
	float value;
    };

    enum PotentialType {
	TYPE,
	NONBONDED,
	BOND,
	SMDBOND,
	ANGLE,
	DIHEDRAL
    };

PotentialSchedule(PotentialType type, size_t index) : potential_type(type), potential_index(index) {}

    
    void add_control_point( size_t step, float value ) {
	size_t s = schedule.size();
	if (s == 0 && step > 1) { // Add initial point if it doesn't exist
	    schedule.push_back( ValueAtStep{1,value} );
	} else if (s > 0) {
	    assert(schedule[s-1].step < step); /* Ensure ordered */
	}
	schedule.push_back( ValueAtStep{step,value} );
	if (s == 0) {
	    next_step = 1;
	    last_idx = 0;
	    slope = 0;
	    intercept = value;
	}
    }
    
    float get_value(size_t step) {
	if (step == 1) {
	    assert(next_step == 1);
	    size_t last_step = next_step;
	    float last_val = schedule[last_idx].value;

	    if (last_idx == schedule.size()-1) {
		// End of schedule; use final value
		float last_val = schedule[last_idx].value;
		intercept = last_val;
		slope = 0;
		next_step = 0;
		return last_val;
	    } else {
		next_step = schedule[last_idx+1].step;
		float next_val = schedule[last_idx+1].value;
		slope = (next_val-last_val)/(next_step-last_step);
		intercept = last_val - last_step*slope;
		return last_val;
	    }
	} else if (next_step == 0) {
	    // End of schedule; use final value
	    return intercept;
	} else if (step < next_step) {
	    // Part of a ramp
	    return (step) * slope + intercept;
	} else if (step == next_step) {
	    // End of ramp segment
	    ++last_idx;
	    if (last_idx == schedule.size()-1) {
		// End of schedule; use final value
		float last_val = schedule[last_idx].value;
		intercept = last_val;
		slope = 0;
		next_step = 0;
		return last_val;
	    } else {
		// New ramp segment
		size_t last_step = next_step;
		float last_val = schedule[last_idx].value;
		
		next_step = schedule[last_idx+1].step;
		float next_val = schedule[last_idx+1].value;
		if (next_val == last_val) {
		    slope = 0.0f;
		} else {
		    slope = (next_val-last_val)/(next_step-last_step);
		}
		intercept = last_val - last_step*slope;
		return last_val;
	    }
	} else {
	    assert(true);
	}	
    }

    PotentialType potential_type;
    size_t potential_index;
    
public:
    std::vector<ValueAtStep> schedule; /* assume this is ordered */

    float slope;
    // float last_val;
    float intercept;
    size_t last_idx;
    // size_t last_step;
    size_t next_step;
};
