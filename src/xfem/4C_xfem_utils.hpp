/*----------------------------------------------------------------------*/
/*! \file
\brief Basic tools used in XFEM routines

\level 3


\warning this file should be cleaned up
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_XFEM_UTILS_HPP
#define FOUR_C_XFEM_UTILS_HPP

#include "4C_config.hpp"

#include "4C_cut_point.hpp"
#include "4C_lib_discret.hpp"

FOUR_C_NAMESPACE_OPEN

namespace XFEM
{
  namespace UTILS
  {
    //! extract the nodal vectors and store them in node-vector-map
    //! \author schott \date 01/13
    void ExtractNodeVectors(Teuchos::RCP<DRT::Discretization> dis,
        std::map<int, CORE::LINALG::Matrix<3, 1>>& nodevecmap, Teuchos::RCP<Epetra_Vector> idispnp);

    //! @name Get material properties for the Volume Cell

    /*!

    \brief Element material for the volume cell, depending on element and position.
           If an element which is not a material list is given, the provided material is chosen.
           If however a material list is given the material chosen for the volume cell is depending
    on the point position.

     */
    void get_volume_cell_material(DRT::Element* actele,  // element for volume cell INPUT
        Teuchos::RCP<CORE::MAT::Material>& mat,          // material of volume cell OUTPUT
        CORE::GEO::CUT::Point::PointPosition position =
            CORE::GEO::CUT::Point::outside  // position of volume cell INPUT to determine
                                            // position
    );


    //! @name Check whether materials are identical
    /*!

    \brief A Safety check is done for XFEM-type problems. Is utilized in the edgebased framework.

     */
    void SafetyCheckMaterials(
        Teuchos::RCP<CORE::MAT::Material>& pmat, Teuchos::RCP<CORE::MAT::Material>& nmat);

    //! @name Extract quantities on a element
    /*!
    \brief Needs a column-vector to extract correctly in parallel
     */
    void ExtractQuantityAtElement(CORE::LINALG::SerialDenseMatrix::Base& element_vector,
        const DRT::Element* element,
        const Teuchos::RCP<const Epetra_MultiVector>& global_col_vector,
        Teuchos::RCP<DRT::Discretization>& dis, const int nds_vector, const int nsd);

    //! @name Extract quantities on a node
    /*!
    \brief Needs a column-vector to extract correctly in parallel
     */
    void ExtractQuantityAtNode(CORE::LINALG::SerialDenseMatrix::Base& element_vector,
        const DRT::Node* node, const Teuchos::RCP<const Epetra_MultiVector>& global_col_vector,
        Teuchos::RCP<DRT::Discretization>& dis, const int nds_vector, const unsigned int nsd);

  }  // namespace UTILS
}  // namespace XFEM


FOUR_C_NAMESPACE_CLOSE

#endif