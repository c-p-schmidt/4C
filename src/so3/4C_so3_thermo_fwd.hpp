// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SO3_THERMO_FWD_HPP
#define FOUR_C_SO3_THERMO_FWD_HPP

/*----------------------------------------------------------------------*/
/*! \file

\level 1

\brief forward declarations of discretisation types

 *
 */

FOUR_C_NAMESPACE_OPEN

// template classes
template class Discret::Elements::So3Thermo<Discret::Elements::SoHex8, Core::FE::CellType::hex8>;
template class Discret::Elements::So3Thermo<Discret::Elements::SoHex8fbar,
    Core::FE::CellType::hex8>;
template class Discret::Elements::So3Thermo<Discret::Elements::SoHex27, Core::FE::CellType::hex27>;
template class Discret::Elements::So3Thermo<Discret::Elements::SoHex20, Core::FE::CellType::hex20>;
template class Discret::Elements::So3Thermo<Discret::Elements::SoTet4, Core::FE::CellType::tet4>;
template class Discret::Elements::So3Thermo<Discret::Elements::SoTet10, Core::FE::CellType::tet10>;
template class Discret::Elements::So3Thermo<Discret::Elements::Nurbs::SoNurbs27,
    Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE

#endif
