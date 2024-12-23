// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_shell7p_ele_factory.hpp"

#include "4C_shell7p_ele.hpp"
#include "4C_shell7p_ele_calc.hpp"
#include "4C_shell7p_ele_calc_eas.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

std::unique_ptr<Discret::Elements::Shell7pEleCalcInterface>
Discret::Elements::Shell7pFactory::provide_shell7p_calculation_interface(
    const Core::Elements::Element& ele, const std::set<Inpar::Solid::EleTech>& eletech)
{
  switch (ele.shape())
  {
    case Core::FE::CellType::quad4:
    {
      return define_calculation_interface_type<Core::FE::CellType::quad4>(eletech);
    }
    case Core::FE::CellType::quad8:
    {
      return define_calculation_interface_type<Core::FE::CellType::quad8>(eletech);
    }
    case Core::FE::CellType::quad9:
    {
      return define_calculation_interface_type<Core::FE::CellType::quad9>(eletech);
    }
    case Core::FE::CellType::tri3:
    {
      return define_calculation_interface_type<Core::FE::CellType::tri3>(eletech);
    }
    case Core::FE::CellType::tri6:
    {
      return define_calculation_interface_type<Core::FE::CellType::tri6>(eletech);
    }
    default:
      FOUR_C_THROW("unknown distype provided");
  }
}

template <Core::FE::CellType distype>
std::unique_ptr<Discret::Elements::Shell7pEleCalcInterface>
Discret::Elements::Shell7pFactory::define_calculation_interface_type(
    const std::set<Inpar::Solid::EleTech>& eletech)
{
  // here we go into the different cases for element technology
  switch (eletech.size())
  {
    // no element technology
    case 0:
      return std::make_unique<Discret::Elements::Shell7pEleCalc<distype>>();
    // simple: just one element technology
    case 1:
      switch (*eletech.begin())
      {
        case Inpar::Solid::EleTech::eas:
        {
          if constexpr ((distype != Core::FE::CellType::quad4) &&
                        (distype != Core::FE::CellType::quad9))
          {
            FOUR_C_THROW("EAS is only implemented for quad4 and quad9 elements.");
          }
          return std::make_unique<Discret::Elements::Shell7pEleCalcEas<distype>>();
        }
        default:
          FOUR_C_THROW("unknown element technology");
      }
    // combination of element technologies
    default:
    {
      FOUR_C_THROW("unknown combination of element technologies.");
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
