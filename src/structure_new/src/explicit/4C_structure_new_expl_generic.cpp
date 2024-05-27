/*-----------------------------------------------------------*/
/*! \file

\brief Generic class for all explicit time integrators


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_expl_generic.hpp"

#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_group_prepostoperator.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_model_evaluator.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::Setup()
{
  check_init();

  // call base class first
  STR::Integrator::Setup();

  // ---------------------------------------------------------------------------
  // set the new pre/post operator for the nox nln group in the parameter list
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p_grp_opt = SDyn().GetNoxParams().sublist("Group Options");

  // create the new generic pre/post operator
  Teuchos::RCP<NOX::NLN::Abstract::PrePostOperator> prepost_generic_ptr =
      Teuchos::rcp(new NOX::NLN::PrePostOp::EXPLICIT::Generic(*this));

  // Get the current map. If there is no map, return a new empty one. (reference)
  NOX::NLN::GROUP::PrePostOperator::Map& prepostgroup_map =
      NOX::NLN::GROUP::PrePostOp::GetMap(p_grp_opt);

  // insert/replace the old pointer in the map
  prepostgroup_map[NOX::NLN::GROUP::prepost_impl_generic] = prepost_generic_ptr;

  // ---------------------------------------------------------------------------
  // set the new pre/post operator for the nox nln solver in the parameter list
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p_sol_opt = SDyn().GetNoxParams().sublist("Solver Options");

  NOX::NLN::AUX::AddToPrePostOpVector(p_sol_opt, prepost_generic_ptr);

  // No issetup_ = true, since the Setup() functions of the derived classes
  // have to be called and finished first!
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::EXPLICIT::Generic::ApplyForce(const Epetra_Vector& x, Epetra_Vector& f)
{
  check_init_setup();

  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  ResetEvalParams();
  bool ok = ModelEval().ApplyForce(x, f, 1.0);
  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::EXPLICIT::Generic::ApplyStiff(const Epetra_Vector& x, CORE::LINALG::SparseOperator& jac)
{
  check_init_setup();

  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  ResetEvalParams();
  const bool ok = ModelEval().ApplyStiff(x, jac, 1.0);

  if (not ok) return ok;

  jac.Complete();

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::EXPLICIT::Generic::ApplyForceStiff(
    const Epetra_Vector& x, Epetra_Vector& f, CORE::LINALG::SparseOperator& jac)
{
  check_init_setup();
  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  ResetEvalParams();
  const bool ok = ModelEval().ApplyForceStiff(x, f, jac, 1.0);

  if (not ok) return ok;

  jac.Complete();

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::EXPLICIT::Generic::CalcRefNormForce(
    const enum ::NOX::Abstract::Vector::NormType& type) const
{
  FOUR_C_THROW("%s is not yet implemented", __FUNCTION__);
  return 0.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::compute_jacobian_contributions_from_element_level_for_ptc(
    Teuchos::RCP<CORE::LINALG::SparseMatrix>& scalingMatrixOpPtr)
{
  FOUR_C_THROW("%s is not yet implemented", __FUNCTION__);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::EXPLICIT::Generic::assemble_force(
    Epetra_Vector& f, const std::vector<INPAR::STR::ModelType>* without_these_models) const
{
  FOUR_C_THROW("%s is not yet implemented", __FUNCTION__);
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::remove_condensed_contributions_from_rhs(Epetra_Vector& rhs) const
{
  ModelEval().remove_condensed_contributions_from_rhs(rhs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::UpdateStepElement()
{
  check_init_setup();
  ModelEval().UpdateStepElement();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::update_constant_state_contributions() {}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::post_update() { update_constant_state_contributions(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::EXPLICIT::Generic::get_default_step_length() const
{
  // default: return a step length of 1.0
  return 1.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::ResetEvalParams()
{
  // set the time step dependent parameters for the element evaluation
  EvalData().SetTotalTime(GlobalState().GetTimeNp());
  EvalData().SetDeltaTime((*GlobalState().GetDeltaTime())[0]);
  EvalData().SetIsTolerateError(true);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::PrePostOp::EXPLICIT::Generic::runPreIterate(const ::NOX::Solver::Generic& nlnSolver)
{
  // For explicit integration this action simply does nothing.
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::PrePostOp::EXPLICIT::Generic::runPreSolve(const ::NOX::Solver::Generic& nlnSolver)
{
  // For explicit integration this action simply does nothing.
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::PrePostOp::EXPLICIT::Generic::runPreComputeX(const NOX::NLN::Group& input_grp,
    const Epetra_Vector& dir, const double& step, const NOX::NLN::Group& curr_grp)
{
  // For explicit integration this action simply does nothing.
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::PrePostOp::EXPLICIT::Generic::runPostComputeX(const NOX::NLN::Group& input_grp,
    const Epetra_Vector& dir, const double& step, const NOX::NLN::Group& curr_grp)
{
  // For explicit integration this action simply does nothing.
}

FOUR_C_NAMESPACE_CLOSE