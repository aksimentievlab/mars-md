/////////////////////////////////////////////////////////////////////////  
// Author: Jeff Comer <jcomer2@illinois.edu> 
#ifndef RANDOMGSL_H
#define RANDOMGSL_H

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "useful.h"

class Random {

private:
  gsl_rng* gslRando;

public:

  // default constructor
  Random() {
    init(0);
  }

  // constructor with seed
  Random(unsigned long seed) {
    init(seed);
  }

  ~Random() {
    gsl_rng_free(gslRando);
  }

  // reinitialize with seed
  void init(unsigned long seed) {
    gslRando = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(gslRando, seed);
  }

  // return a number uniformly distributed between 0 and 1
  float uniform() {
    return gsl_rng_uniform(gslRando);
  }

  long poisson(float lambda) {
     return gsl_ran_poisson(gslRando, lambda);
  }

  // return a number from a standard gaussian distribution
  float gaussian() {
    return gsl_ran_ugaussian(gslRando);
  }

  // return a vector of gaussian random numbers
  Vector3 gaussian_vector() {
    return Vector3( gaussian(), gaussian(), gaussian() );
  }

  // return a random long
  long integer() {
    return gsl_rng_get(gslRando);
  }

  // randomly order an array of whatever
  template <class Elem> void reorder(Elem *a, int n) {
    for (int i = 0; i < (n-1); i++) {
      int ie = i + (integer()%(n-i));
      if (ie == i) continue;
      // Swap.
      const Elem e = a[ie];
      a[ie] = a[i];
      a[i] = e;
    }
  }
};

#endif


