// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_DIRECTION_FACTORY_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_DIRECTION_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_forward_decl.hpp"

#include <NOX_Direction_UserDefinedFactory.H>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace Direction
    {
      /*!
       \brief Factory to build direction objects derived from ::NOX::Direction::Generic.

       This factory class is closely related to the NOX_Direction_Factory.H. The main
       difference is that it allows to use direction methods which differ from the
       default NOX package.
       */
      class Factory : public ::NOX::Direction::UserDefinedFactory
      {
       public:
        //! Constructor
        Factory();

        /*! \brief Factory to build user-defined direction objects.

            \note All direction methods which differ from the default NOX direction
            routines are user-defined. That means the user is the 4C user and the 4C
            programmer.

            @param gd A global data pointer that contains the top level parameter list.  Without
           storing this inside the direction object, there is no guarantee that the second parameter
           \c params will still exist.  It can be deleted by the top level RCP.
            @param params Sublist with direction construction parameters.

        */
        Teuchos::RCP<::NOX::Direction::Generic> buildDirection(
            const Teuchos::RCP<::NOX::GlobalData>& gd,
            Teuchos::ParameterList& params) const override;
      };

      /*! Nonmember function to build a direction object.

      \relates NOX::Nln::Direction::Factory

      */
      Teuchos::RCP<::NOX::Direction::Generic> build_direction(
          const Teuchos::RCP<::NOX::GlobalData>& gd, Teuchos::ParameterList& params);

    }  // namespace Direction
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
