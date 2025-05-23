// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_NURBS_DISCRETIZATION_UTILS_HPP
#define FOUR_C_FEM_NURBS_DISCRETIZATION_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  namespace Nurbs
  {
    /*----------------------------------------------------------------------*/
    /*!
    \brief A service method for accessing knotvector and weights for
           an isogeometric element


    \param dis         (i) the discretisation
    \param ele         (i) a pointer to the element
    \param myknots     (o) knot vector (to be filled)
    \param weights     (o) weight vector (to be filled)


    */
    template <class WG>
    bool get_my_nurbs_knots_and_weights(const Core::FE::Discretization& dis,
        const Core::Elements::Element* ele, std::vector<Core::LinAlg::SerialDenseVector>& myknots,
        WG& weights)
    {
      // try to cast dis to a nurbs discretisation
      const Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
          dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&(dis));
      if (nurbsdis == nullptr) FOUR_C_THROW("Received discretization which is not Nurbs!");

      // get local knot vector entries and check for zero sized elements
      const bool zero_size = (*((*nurbsdis).get_knot_vector())).get_ele_knots(myknots, ele->id());

      // if we have a zero sized element due to a interpolated
      // point --- exit here and tell the outside world about that
      if (zero_size)
      {
        return (zero_size);
      }
      // you are still here? So get the node weights for the nurbs element as well
      const Core::Nodes::Node* const* nodes = ele->nodes();
      const int nen = ele->num_node();
      for (int inode = 0; inode < nen; inode++)
      {
        const Core::FE::Nurbs::ControlPoint* cp =
            dynamic_cast<const Core::FE::Nurbs::ControlPoint*>(nodes[inode]);
        weights(inode) = cp->w();
      }

      // goodbye
      return zero_size;
    }  // get_my_nurbs_knots_and_weights()

    /*!
    \brief A service method for accessing knotvector and weights for
           an isogeometric boundary element


    \param boundaryele      (i) a pointer to the boundary element
    \param localsurfaceid   (i) local id of this boundary element
    \param parenteleid      (i) global id of parent element
    \param dis              (i) the discretisation
    \param myknots          (o) parent knot vector (to be filled)
    \param myknots          (o) knot vector for boundary element (to be filled)
    \param weights          (o) weight vector for boundary element (to be filled)
    \param normalfac        (o) normalfac (to be filled)


    */
    template <class WG>
    bool get_knot_vector_and_weights_for_nurbs_boundary(const Core::Elements::Element* boundaryele,
        const int localsurfaceid, const int parenteleid,
        const Core::FE::Discretization& discretization,
        std::vector<Core::LinAlg::SerialDenseVector>& mypknots,
        std::vector<Core::LinAlg::SerialDenseVector>& myknots, WG& weights, double& normalfac)
    {
      // get knotvector(s)
      const Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
          dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&(discretization));

      std::shared_ptr<const Core::FE::Nurbs::Knotvector> knots = (*nurbsdis).get_knot_vector();

      bool zero_size = knots->get_boundary_ele_and_parent_knots(
          mypknots, myknots, normalfac, parenteleid, localsurfaceid);

      // if we have a zero sized element due to a interpolated
      // point --- exit here and tell the outside world about that
      if (zero_size)
      {
        return (zero_size);
      }
      // you are still here? So get the node weights as well
      const Core::Nodes::Node* const* nodes = boundaryele->nodes();
      const int boundarynen = boundaryele->num_node();
      for (int inode = 0; inode < boundarynen; inode++)
      {
        const Core::FE::Nurbs::ControlPoint* cp =
            dynamic_cast<const Core::FE::Nurbs::ControlPoint*>(nodes[inode]);
        weights(inode) = cp->w();
      }

      // goodbye
      return zero_size;
    }  // get_knot_vector_and_weights_for_nurbs_boundary()

    /*!
    \brief A service method for accessing knotvector and weights for
           an isogeometric boundary element and parent element


    \param boundaryele      (i) a pointer to the boundary element
    \param localsurfaceid   (i) local id of this boundary element
    \param parenteleid      (i) global id of parent element
    \param dis              (i) the discretisation
    \param pmyknots         (o) parent knot vector (to be filled)
    \param myknots          (o) knot vector for boundary element (to be filled)
    \param pweights         (o) weight vector for parent element (to be filled)
    \param weights          (o) weight vector for boundary element (to be filled)
    \param normalfac        (o) normalfac (to be filled)


    */
    template <class WG>
    bool get_knot_vector_and_weights_for_nurbs_boundary_and_parent(
        Core::Elements::Element* parentele, Core::Elements::Element* boundaryele,
        const int localsurfaceid, const Core::FE::Discretization& discretization,
        std::vector<Core::LinAlg::SerialDenseVector>& mypknots,
        std::vector<Core::LinAlg::SerialDenseVector>& myknots, WG& pweights, WG& weights,
        double& normalfac)
    {
      // get knotvector(s)
      const Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
          dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&(discretization));

      std::shared_ptr<const Core::FE::Nurbs::Knotvector> knots = (*nurbsdis).get_knot_vector();

      bool zero_size = knots->get_boundary_ele_and_parent_knots(
          mypknots, myknots, normalfac, parentele->id(), localsurfaceid);

      // if we have a zero sized element due to a interpolated
      // point --- exit here and tell the outside world about that
      if (zero_size)
      {
        return (zero_size);
      }
      // you are still here? So get the node weights as well
      Core::Nodes::Node** nodes = boundaryele->nodes();
      const int boundarynen = boundaryele->num_node();
      for (int inode = 0; inode < boundarynen; inode++)
      {
        Core::FE::Nurbs::ControlPoint* cp =
            dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes[inode]);
        weights(inode) = cp->w();
      }

      Core::Nodes::Node** pnodes = parentele->nodes();
      const int pnen = parentele->num_node();
      for (int inode = 0; inode < pnen; inode++)
      {
        Core::FE::Nurbs::ControlPoint* cp =
            dynamic_cast<Core::FE::Nurbs::ControlPoint*>(pnodes[inode]);
        pweights(inode) = cp->w();
      }

      // goodbye
      return zero_size;
    }  // get_knot_vector_and_weights_for_nurbs_boundary()

    /**
     * \brief Helper function to evaluate the NURBS interpolation inside the element.

     \param controlpoint_data   (in) data from control points
     \param xi                  (in) Parameter coordinate on the element.
     \param weights             (in) weights of the control points
     \param knots               (in) knots where the NURBS element is defined
     \param distype             (in) discretization type of NURBS element
     */
    template <unsigned int n_points, unsigned int n_dim_nurbs, unsigned int n_dim = n_dim_nurbs>
    Core::LinAlg::Matrix<n_dim, 1, double> eval_nurbs_interpolation(
        const Core::LinAlg::Matrix<n_points * n_dim, 1, double>& controlpoint_data,
        const Core::LinAlg::Matrix<n_dim_nurbs, 1, double>& xi,
        const Core::LinAlg::Matrix<n_points, 1, double>& weights,
        const std::vector<Core::LinAlg::SerialDenseVector>& knots,
        const Core::FE::CellType& distype)
    {
      Core::LinAlg::Matrix<n_dim, 1, double> point_result;

      // Get the shape functions.
      Core::LinAlg::Matrix<n_points, 1, double> N;

      if (n_dim_nurbs == 3)
        nurbs_get_3d_funct(N, xi, knots, weights, distype);
      else if (n_dim_nurbs == 2)
        nurbs_get_2d_funct(N, xi, knots, weights, distype);
      else
        FOUR_C_THROW("Unable to compute the shape functions for this nurbs element case");

      for (unsigned int i_node_nurbs = 0; i_node_nurbs < n_points; i_node_nurbs++)
      {
        for (unsigned int i_dim = 0; i_dim < n_dim; i_dim++)
          point_result(i_dim) += N(i_node_nurbs) * controlpoint_data(i_node_nurbs * n_dim + i_dim);
      }

      return point_result;
    }

  }  // namespace Nurbs

}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
