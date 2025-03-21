// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_timint_poro_genalpha.hpp"

#include "4C_io.hpp"

FOUR_C_NAMESPACE_OPEN


FLD::TimIntPoroGenAlpha::TimIntPoroGenAlpha(const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis, solver, params, output, alefluid),
      TimIntGenAlpha(actdis, solver, params, output, alefluid),
      TimIntPoro(actdis, solver, params, output, alefluid)
{
}


void FLD::TimIntPoroGenAlpha::init()
{
  // call init()-functions of base classes
  // note: this order is important
  TimIntGenAlpha::init();
  TimIntPoro::init();
}

void FLD::TimIntPoroGenAlpha::gen_alpha_intermediate_values()
{
  // set intermediate values for acceleration and potential temporal
  // derivatives
  //
  //       n+alphaM                n+1                      n
  //    acc         = alpha_M * acc     + (1-alpha_M) *  acc
  //       (i)                     (i)

  // consider both velocity and pressure degrees of freedom
  accam_->update((alphaM_), *accnp_, (1.0 - alphaM_), *accn_, 0.0);

  // set intermediate values for velocity
  //
  //       n+alphaF              n+1                   n
  //      u         = alpha_F * u     + (1-alpha_F) * u
  //       (i)                   (i)
  //
  // and pressure
  //
  //       n+alphaF              n+1                   n
  //      p         = alpha_F * p     + (1-alpha_F) * p
  //       (i)                   (i)
  //
  // note that its af-genalpha with mid-point treatment of the pressure,
  // not implicit treatment as for the genalpha according to Whiting
  velaf_->update((alphaF_), *velnp_, (1.0 - alphaF_), *veln_, 0.0);
}

void FLD::TimIntPoroGenAlpha::read_restart(int step)
{
  // call of base classes
  TimIntGenAlpha::read_restart(step);
  TimIntPoro::read_restart(step);
}

FOUR_C_NAMESPACE_CLOSE
