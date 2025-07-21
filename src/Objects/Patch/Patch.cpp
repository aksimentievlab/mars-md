#include "Patch.h"
#include "PatchOp.h"

namespace ARBD {
void Patch::compute() {
  for (auto &op : ops) {
    switch (op.num_patches()) {
    case 1:
      op.compute(this) break;
    case 2:
      for (auto neighbor : neighbor_patches) {
        Patch *patch_p[2];
        patch_p[0] = this;
        patch_p[1] = neighbor;
        op.compute(patch_p);
      }
      break;
    default:
      std::cerr << "Error: Patch::compute: "
                << "PatchOp operates on unhandled number of patches; exiting"
                << std::endl;
      assert(false);
    }
  }
};
} // namespace ARBD
