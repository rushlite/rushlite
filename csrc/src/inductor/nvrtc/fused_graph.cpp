#include "lamp3/inductor/nvrtc/fused_graph.hpp"

#include "lamp3/common/assert.hpp"

namespace lmp::inductor {
namespace {

void visit(tensor::TensorImpl* n, FusedGraph& g) {
  if (g.seen.count(n))
    return;
  g.seen.insert(n);

  if (!n->is_deferred() || !n->lazy_op()->is_fusible()) {
    tensor::lazy::realize(n);
    LMP_INTERNAL_ASSERT(n->is_contiguous())
        << "Generated fusion kernels require contiguous input leaves";
    if (!g.slot.count(n)) {
      g.slot[n] = g.inputs.size();
      g.inputs.push_back(n);
    }
    return;
  }

  for (const std::shared_ptr<tensor::TensorImpl>& in : n->lazy_op()->inputs)
    visit(in.get(), g);
  g.order.push_back(n);
}

}

FusedGraph build_fused_graph(tensor::TensorImpl* root) {
  FusedGraph g;
  g.output = root;
  g.seen.insert(root);
  for (const std::shared_ptr<tensor::TensorImpl>& in : root->lazy_op()->inputs)
    visit(in.get(), g);
  g.order.push_back(root);
  return g;
}

}
