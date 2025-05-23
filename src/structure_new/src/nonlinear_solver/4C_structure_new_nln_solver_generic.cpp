// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_nln_solver_generic.hpp"

#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_implicit.hpp"
#include "4C_structure_new_timint_noxinterface.hpp"

#include <NOX_Abstract_Group.H>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Nln::SOLVER::Generic::Generic()
    : isinit_(false),
      issetup_(false),
      gstate_ptr_(nullptr),
      sdyn_ptr_(nullptr),
      noxinterface_ptr_(nullptr),
      group_ptr_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::SOLVER::Generic::init(
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate,
    const std::shared_ptr<Solid::TimeInt::BaseDataSDyn>& sdyn,
    const std::shared_ptr<Solid::TimeInt::NoxInterface>& noxinterface,
    const std::shared_ptr<Solid::Integrator>& integrator,
    const std::shared_ptr<const Solid::TimeInt::Base>& timint)
{
  // We have to call setup() after init()
  issetup_ = false;

  // initialize internal variables
  gstate_ptr_ = gstate;
  sdyn_ptr_ = sdyn;
  noxinterface_ptr_ = noxinterface;
  int_ptr_ = integrator;
  timint_ptr_ = timint;

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Abstract::Group>& Solid::Nln::SOLVER::Generic::group_ptr()
{
  check_init();

  return group_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
::NOX::Abstract::Group& Solid::Nln::SOLVER::Generic::get_solution_group()
{
  check_init_setup();
  FOUR_C_ASSERT(group_ptr_, "The group pointer should be initialized beforehand!");

  return *group_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const ::NOX::Abstract::Group& Solid::Nln::SOLVER::Generic::get_solution_group() const
{
  check_init_setup();
  FOUR_C_ASSERT(group_ptr_, "The group pointer should be initialized beforehand!");

  return *group_ptr_;
}

FOUR_C_NAMESPACE_CLOSE
