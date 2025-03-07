// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_TSI_UTILS_HPP
#define FOUR_C_TSI_UTILS_HPP


#include "4C_config.hpp"

#include "4C_coupling_volmortar_utils.hpp"

#include <mpi.h>

#include <map>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

/*----------------------------------------------------------------------*
 |                                                           dano 11/09 |
 *----------------------------------------------------------------------*/
namespace TSI
{
  namespace Utils
  {
    //! \brief implementation of special clone strategy for automatic generation
    //!        of thermo discretization from a given structure discretization
    class ThermoStructureCloneStrategy
    {
     public:
      //! constructor
      explicit ThermoStructureCloneStrategy() {}
      //! destructor
      virtual ~ThermoStructureCloneStrategy() = default;

     protected:
      //! determine element type std::string and whether element is copied or not
      bool determine_ele_type(
          Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype);

      //! set element-specific data (material etc.)
      void set_element_data(std::shared_ptr<Core::Elements::Element> newele,
          Core::Elements::Element* oldele, const int matid, const bool isnurbs);

      //! returns conditions names to be copied (source and target name)
      std::map<std::string, std::string> conditions_to_copy() const;

      //! check for correct material
      void check_material_type(const int matid);

     private:
    };  // class ThermoStructureCloneStrategy

    //! setup TSI, clone the structural discretization
    void setup_tsi(MPI_Comm comm);


    void set_material_pointers_matching_grid(
        const Core::FE::Discretization& sourcedis, const Core::FE::Discretization& targetdis);

    //! strategy for material assignment for non matching meshes with TSI

    /// Helper class for assigning materials for volumetric coupling of non conforming meshes (TSI)
    /*!
     When coupling two overlapping discretizations, most often one discretization needs access
     to the corresponding element/material on the other side. For conforming meshes this is straight
     forward as there is one unique element on the other side and therefore one unique material,
     which can be accessed. However, for non conforming meshes there are potentially several
     elements overlapping. Therefore, some rule for assigning materials is needed. This class is
     meant to do that. It gets the element to which it shall assign a material and a vector of IDs
     of the overlapping elements of the other discretization.

     In case of TSI, we also need the kinematic type of the structural element to be known in the
     thermo element, which is why there is a special strategy for TSI. Note that this is not yet
     working for inhomogeneous material properties.

     \author vuong 10/14
     */
    class TSIMaterialStrategy : public Coupling::VolMortar::Utils::DefaultMaterialStrategy
    {
     public:
      //! constructor
      TSIMaterialStrategy() {};

      //! assignment of thermo material to structure material
      void assign_material2_to1(const Coupling::VolMortar::VolMortarCoupl* volmortar,
          Core::Elements::Element* ele1, const std::vector<int>& ids_2,
          std::shared_ptr<Core::FE::Discretization> dis1,
          std::shared_ptr<Core::FE::Discretization> dis2) override;

      //! assignment of structure material to thermo material
      void assign_material1_to2(const Coupling::VolMortar::VolMortarCoupl* volmortar,
          Core::Elements::Element* ele2, const std::vector<int>& ids_1,
          std::shared_ptr<Core::FE::Discretization> dis1,
          std::shared_ptr<Core::FE::Discretization> dis2) override;
    };

  }  // namespace Utils

  //! prints the 4C tsi-logo on the screen
  void printlogo();

}  // namespace TSI


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
