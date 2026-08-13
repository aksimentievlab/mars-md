#!/bin/bash
#
# autogen-mkl.sh -- regenerate autotools files and configure scuff-em
#                    to build against Intel oneMKL instead of the
#                    system BLAS/LAPACK.
#
# Usage:
#   source /opt/intel/oneapi/setvars.sh        # or wherever your oneMKL lives
#   ./autogen-mkl.sh                            # optionally pass extra
#                                                #   ./configure flags after this script's own args
#
# Environment overrides:
#   MKLROOT       - path to oneMKL install (required; set by setvars.sh)
#   MKL_THREADING - "sequential" (default) or "gnu_thread"
#                   sequential = MKL does no internal threading itself;
#                     safest choice since scuff-em's own OpenMP loops
#                     (compiled against libgomp) are already doing the
#                     parallelism you want, and mixing libiomp5 (Intel
#                     OpenMP, used by mkl_intel_thread) with libgomp in
#                     one process is a known source of oversubscription/
#                     hangs.
#                   gnu_thread = lets MKL also thread internally
#                     (e.g. the zgetrf/zgetrs call itself), linked
#                     against libgomp so it shares scuff-em's runtime
#                     instead of pulling in libiomp5.
#sed -i '35a #include <stdlib.h>' libs/libSpherical/machcon.c
set -euo pipefail
mkdir -p build/scuff-link
SCUFF_PREFIX="$(pwd -P)/build/scuff-link"
cd extern/scuff-em
# --- sanity checks -------------------------------------------------------
if [ -z "${MKLROOT:-}" ]; then
  echo "error: MKLROOT is not set." >&2
  echo "  Run e.g. 'source /opt/intel/oneapi/setvars.sh' or" >&2
  echo "  'source /opt/intel/oneapi/mkl/latest/env/vars.sh' first." >&2
  exit 1
fi

MKL_LIBDIR="$MKLROOT/lib/intel64"
[ -d "$MKL_LIBDIR" ] || MKL_LIBDIR="$MKLROOT/lib"   # newer oneMKL layouts drop /intel64
if [ ! -d "$MKL_LIBDIR" ]; then
  echo "error: could not find MKL library directory under $MKLROOT" >&2
  exit 1
fi

MKL_THREADING="${MKL_THREADING:-sequential}"
case "$MKL_THREADING" in
  sequential) MKL_THREAD_LIB="mkl_sequential" ;;
  gnu_thread) MKL_THREAD_LIB="mkl_gnu_thread" ;;
  *) echo "error: MKL_THREADING must be 'sequential' or 'gnu_thread'" >&2; exit 1 ;;
esac


# LP64 (32-bit int) interface -- required, since libs/libhmat/lapack.h
# declares plain `int` arguments. Do NOT use the ilp64 libs here.
MKL_LINK="-Wl,--start-group -L${MKL_LIBDIR} -lmkl_intel_lp64 -l${MKL_THREAD_LIB} -lmkl_core -Wl,--end-group -lpthread -lm -ldl"

echo "== Building scuff-em against oneMKL =="
echo "   MKLROOT       = $MKLROOT"
echo "   MKL lib dir   = $MKL_LIBDIR"
echo "   threading     = $MKL_THREADING"
echo "   install prefix= $SCUFF_PREFIX"
echo

# --- regenerate autotools files ------------------------------------------
touch ChangeLog
autoreconf --verbose --install --symlink --force

# --- configure ------------------------------------------------------------
# Both BLAS_LIBS/LAPACK_LIBS (env vars honored directly by ACX_BLAS/
# ACX_LAPACK, bypassing their autodetection probes) and --with-blas=/
# --with-lapack= are set, since exact option names can differ slightly
# between versions of these macros -- belt and suspenders.
BLAS_LIBS="$MKL_LINK" \
LAPACK_LIBS="$MKL_LINK" \
./configure \
  --enable-maintainer-mode \
  --prefix="$SCUFF_PREFIX" \
  --with-blas="$MKL_LINK" \
  --with-lapack="$MKL_LINK" \
  --without-hdf5 \
  LDFLAGS="-L${MKL_LIBDIR} -Wl,-rpath,${MKL_LIBDIR} ${LDFLAGS:-}" \
  "$@"
make -j 20
make install

echo
echo "== Configure finished. Sanity-checking what got picked up: =="
grep -i -A2 "checking for.*[dz]getrf" config.log || true
echo
echo "If the lines above don't mention mkl_intel_lp64, check config.log"
echo "directly and/or run './configure --help | grep -i blas' to confirm"
echo "the option names ACX_BLAS/ACX_LAPACK expect in this tree's m4/ macros."
echo
echo "After building, verify with:"
echo "  ldd $SCUFF_PREFIX/lib/libscuff.so | grep mkl"
echo "  nm -D $SCUFF_PREFIX/lib/libscuff.so | grep -w zgetrf_"
