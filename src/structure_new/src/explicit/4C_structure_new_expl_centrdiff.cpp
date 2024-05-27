/*-----------------------------------------------------------*/
/*! \file

\brief Central differences time integration for solid dynamics


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_expl_centrdiff.hpp"

#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_structure_new_model_evaluator.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::EXPLICIT::CentrDiff::CentrDiff()
    : fvisconp_ptr_(Teuchos::null),
      fviscon_ptr_(Teuchos::null),
      finertianp_ptr_(Teuchos::null),
      finertian_ptr_(Teuchos::null)
{
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::CentrDiff::Setup()
{
  check_init();

  // Call the Setup() of the abstract base class first.
  Generic::Setup();

  // ---------------------------------------------------------------------------
  // setup pointers to the force vectors of the global state data container
  // ---------------------------------------------------------------------------
  finertian_ptr_ = GlobalState().GetFinertialN();
  finertianp_ptr_ = GlobalState().GetFinertialNp();

  fviscon_ptr_ = GlobalState().GetFviscoN();
  fvisconp_ptr_ = GlobalState().GetFviscoNp();

  // -------------------------------------------------------------------
  // set initial displacement
  // -------------------------------------------------------------------
  set_initial_displacement(
      TimInt().GetDataSDyn().GetInitialDisp(), TimInt().GetDataSDyn().StartFuncNo());

  // Has to be set before the post_setup() routine is called!
  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::CentrDiff::post_setup()
{
  check_init_setup();
  equilibrate_initial_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::CentrDiff::set_state(const Epetra_Vector& x)
{
  check_init_setup();

  const double dt = (*GlobalState().GetDeltaTime())[0];
  const double dthalf = dt / 2.0;

  ModelEval().ResetStepState();

  // ---------------------------------------------------------------------------
  // new end-point acceleration
  // ---------------------------------------------------------------------------
  Teuchos::RCP<Epetra_Vector> accnp_ptr = GlobalState().ExtractDisplEntries(x);
  GlobalState().GetAccNp()->Scale(1.0, *accnp_ptr);

  // ---------------------------------------------------------------------------
  // new half-point velocities
  // ---------------------------------------------------------------------------
  GlobalState().GetVelNp()->Update(1.0, *GlobalState().GetVelN(), 0.0);
  GlobalState().GetVelNp()->Update(dthalf, *GlobalState().GetAccN(), 1.0);

  // ---------------------------------------------------------------------------
  // new end-point displacements
  // ---------------------------------------------------------------------------
  GlobalState().GetDisNp()->Update(1.0, *GlobalState().GetDisN(), 0.0);
  GlobalState().GetDisNp()->Update(dt, *GlobalState().GetVelNp(), 1.0);

  // ---------------------------------------------------------------------------
  // update the elemental state
  // ---------------------------------------------------------------------------
  ModelEval().UpdateResidual();
  ModelEval().RunRecover();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::CentrDiff::add_visco_mass_contributions(Epetra_Vector& f) const
{
  // viscous damping forces at t_{n+1}
  CORE::LINALG::AssembleMyVector(1.0, f, 1.0, *fvisconp_ptr_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::CentrDiff::add_visco_mass_contributions(CORE::LINALG::SparseOperator& jac) const
{
  Teuchos::RCP<CORE::LINALG::SparseMatrix> stiff_ptr = GlobalState().ExtractDisplBlock(jac);
  // set mass matrix
  stiff_ptr->Add(*GlobalState().GetMassMatrix(), false, 1.0, 0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::CentrDiff::WriteRestart(
    IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  check_init_setup();
  // write dynamic forces
  iowriter.WriteVector("finert", finertian_ptr_);
  iowriter.WriteVector("fvisco", fviscon_ptr_);

  ModelEval().WriteRestart(iowriter, forced_writerestart);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::CentrDiff::read_restart(IO::DiscretizationReader& ioreader)
{
  check_init_setup();
  ioreader.ReadVector(finertian_ptr_, "finert");
  ioreader.ReadVector(fviscon_ptr_, "fvisco");

  ModelEval().read_restart(ioreader);
  update_constant_state_contributions();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::CentrDiff::UpdateStepState()
{
  check_init_setup();

  const double dt = (*GlobalState().GetDeltaTime())[0];
  const double dthalf = dt / 2.0;

  // ---------------------------------------------------------------------------
  // dynamic effects
  // ---------------------------------------------------------------------------
  // new at t_{n+1} -> t_n
  //    finertial_{n} := finertial_{n+1}
  finertian_ptr_->Scale(1.0, *finertianp_ptr_);
  // new at t_{n+1} -> t_n
  //    fviscous_{n} := fviscous_{n+1}
  fviscon_ptr_->Scale(1.0, *fvisconp_ptr_);

  // recompute the velocity to account for new acceleration
  GlobalState().GetVelNp()->Update(1.0, *GlobalState().GetVelN(), 0.0);
  GlobalState().GetVelNp()->Update(dthalf, *GlobalState().GetAccN(), 1.0);
  GlobalState().GetVelNp()->Update(dthalf, *GlobalState().GetAccNp(), 1.0);

  // ---------------------------------------------------------------------------
  // update model specific variables
  // ---------------------------------------------------------------------------
  ModelEval().UpdateStepState(0.0);
}

FOUR_C_NAMESPACE_CLOSE