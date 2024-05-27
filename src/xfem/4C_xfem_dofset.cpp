/*----------------------------------------------------------------------*/
/*! \file

\brief provides a general XFEM dofset which uses the information from the cut-library to determine
the number of dofs per node when multiple sets of degrees of freedom per node have to be used


\level 1

*/
/*----------------------------------------------------------------------*/

#include "4C_xfem_dofset.hpp"

#include "4C_cut_cutwizard.hpp"
#include "4C_cut_node.hpp"

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  Get the gid of all dofs of a node                      schott 12/14 |
 *----------------------------------------------------------------------*/
void XFEM::XFEMDofSet::Dof(
    std::vector<int>& dofs, const DRT::Node* node, unsigned nodal_dofset_id) const
{
  const int lid = node->LID();
  if (lid == -1) return;
  int numdf = CORE::Dofsets::DofSet::NumDofPerNode(*node);
  const int idx = (*idxcolnodes_)[lid] + nodal_dofset_id * numdf;
  dofs.reserve(numdf);
  for (int i = 0; i < numdf; ++i)
  {
    dofs.push_back(idx + i);
  }
}

/*----------------------------------------------------------------------*
 |  Get the gid of all dofs of a node                      schott 12/14 |
 *----------------------------------------------------------------------*/
int XFEM::XFEMDofSet::NumDofPerNode(const DRT::Node& node) const
{
  CORE::GEO::CUT::Node* n = wizard_.GetNode(node.Id());
  if (n != nullptr)
  {
    int numdofpernode = CORE::Dofsets::DofSet::NumDofPerNode(node);
    return numdofpernode * n->NumDofSets();
  }
  return CORE::Dofsets::DofSet::NumDofPerNode(node);
}

FOUR_C_NAMESPACE_CLOSE