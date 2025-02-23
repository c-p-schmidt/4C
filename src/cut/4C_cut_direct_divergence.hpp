// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_DIRECT_DIVERGENCE_HPP
#define FOUR_C_CUT_DIRECT_DIVERGENCE_HPP

#include "4C_config.hpp"

#include "4C_cut_element.hpp"
#include "4C_cut_volumecell.hpp"

// Choose whether to output divirgence cell information in global coordinates (for easier comparison
// to Tesselation).
#define OUTPUT_GLOBAL_DIVERGENCE_CELLS

FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  class Element;
  class Facet;
  class VolumeCell;
  class Mesh;

  /*!
  \brief A class to construct Gaussian rule for volumecell by direct application of divergence
  theorem. This generate only the integration points on the facets.
  */
  class DirectDivergence
  {
   public:
    DirectDivergence(
        VolumeCell* volcell, Element* elem, const Cut::Point::PointPosition posi, Mesh& mesh)
        : volcell_(volcell), elem1_(elem), position_(posi), mesh_(mesh)
    {
    }

    /*!
    \brief Generate integration points on the facets of the volumecell
    */
    std::shared_ptr<Core::FE::GaussPoints> vc_integration_rule(std::vector<double>& RefPlaneEqn);

    /*!
    \brief Compute and set correspondingly the volume of the considered volumecell from the
    generated integration rule and compare it with full application of divergence theorem
     */
    void debug_volume(const Core::FE::GaussIntegration& gpv, bool& isNeg);

    /*!
    \brief Geometry of volumecell, reference facet, main and internal gauss points for gmsh
    output.
     */
    void divengence_cells_gmsh(
        const Core::FE::GaussIntegration& gpv, Core::FE::GaussPoints& gpmain);

   private:
    /*!
    \brief Identify the list of facets which need to be triangulated, and also get the reference
    facet that will be used in xfluid part
     */
    void list_facets(std::vector<plain_facet_set::const_iterator>& facetIterator,
        std::vector<double>& RefPlaneEqn, plain_facet_set::const_iterator& IteratorRefFacet,
        bool& IsRefFacet);



    //! volumecell over which we construct integration scheme
    VolumeCell* volcell_;

    //! background element that contains this volumecell
    Element* elem1_;

    //! position of this volumecell
    const Cut::Point::PointPosition position_;

    //! mesh that contains the background element
    Mesh& mesh_;

    //! reference facet identified for this volumecell
    Facet* ref_facet_;

    //! true if the reference plane is on a facet of volumecell
    bool is_ref_;

    //! Points that define the reference plane used for this volumecell
    std::vector<Point*> ref_pts_gmsh_;
  };
}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
