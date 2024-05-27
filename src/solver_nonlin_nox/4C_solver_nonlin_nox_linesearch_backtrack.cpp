/*-----------------------------------------------------------*/
/*! \file

\brief %NOX::NLN backtracking line search implementation.



\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_solver_nonlin_nox_linesearch_backtrack.hpp"  // class definition

#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_linesearch_prepostoperator.hpp"
#include "4C_solver_nonlin_nox_solver_linesearchbased.hpp"
#include "4C_solver_nonlin_nox_statustest_normf.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_Vector.h>
#include <NOX_GlobalData.H>
#include <NOX_Utils.H>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::LineSearch::Backtrack::Backtrack(const Teuchos::RCP<::NOX::GlobalData>& gd,
    const Teuchos::RCP<::NOX::StatusTest::Generic> outerTests,
    const Teuchos::RCP<NOX::NLN::INNER::StatusTest::Generic> innerTests,
    Teuchos::ParameterList& params)
    : ls_iters_(0),
      step_ptr_(nullptr),
      default_step_(0.0),
      reduction_factor_(0.0),
      check_type_(::NOX::StatusTest::Complete),
      status_(NOX::NLN::INNER::StatusTest::status_unevaluated),
      outer_tests_ptr_(outerTests),
      inner_tests_ptr_(innerTests)
{
  reset(gd, params);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool NOX::NLN::LineSearch::Backtrack::reset(
    const Teuchos::RCP<::NOX::GlobalData>& gd, Teuchos::ParameterList& params)
{
  Teuchos::ParameterList& p = params.sublist("Backtrack");

  utils_ = gd->getUtils();
  merit_function_ptr_ = gd->getMeritFunction();

  ls_iters_ = 0;
  search_direction_ptr_ = Teuchos::null;

  status_ = NOX::NLN::INNER::StatusTest::status_unevaluated;

  default_step_ = p.get("Default Step", 1.0);
  reduction_factor_ = p.get("Reduction Factor", 0.5);
  if ((reduction_factor_ <= 0.0) || (reduction_factor_ >= 1.0))
  {
    std::ostringstream msg;
    msg << "Invalid choice \"" << reduction_factor_ << "\" for \"Reduction Factor\"!\n"
        << "Value must be greater than zero and less than 1.0.";
    throw_error("reset", msg.str());
  }

  check_type_ = Teuchos::getIntegralValue<::NOX::StatusTest::CheckType>(
      params, "Inner Status Test Check Type");

  fp_except_.shall_be_caught_ = p.get("Allow Exceptions", false);

  prePostOperatorPtr_ = Teuchos::rcp(new PrePostOperator(params));

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::LineSearch::Backtrack::reset()
{
  ls_iters_ = 0;
  search_direction_ptr_ = Teuchos::null;

  status_ = NOX::NLN::INNER::StatusTest::status_unevaluated;

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool NOX::NLN::LineSearch::Backtrack::compute(::NOX::Abstract::Group& grp, double& step,
    const ::NOX::Abstract::Vector& dir, const ::NOX::Solver::Generic& s)
{
  fp_except_.precompute();
  // -------------------------------------------------
  // (re)set important line search parameters
  // -------------------------------------------------
  reset();
  // get the old solution group
  const ::NOX::Abstract::Group& oldGrp = s.getPreviousSolutionGroup();
  // update the search direction pointer
  search_direction_ptr_ = Teuchos::rcpFromRef(dir);
  // set the step pointer to the inserted step variable
  step_ptr_ = &step;
  // reset the step length
  step = default_step_;
  // initialize the inner status test
  // ----------------------------------------------------------------------
  // BE CAREFUL HERE:
  // During the copy operation in Solver::LineSearchBased::step()
  // the current grp loses the ownership of the sharedLinearSystem. If we
  // want to access the jacobian, we have to use the oldGrp
  // (target of the copy process), instead.                hiermeier 08/15
  // ----------------------------------------------------------------------
  const ::NOX::Epetra::Group* epetraOldGrpPtr = dynamic_cast<const ::NOX::Epetra::Group*>(&oldGrp);
  if (not epetraOldGrpPtr->isJacobian())
    throw_error("compute()", "Ownership changed unexpectedly!");

  /* Setup the inner status test */
  status_ = inner_tests_ptr_->CheckStatus(*this, s, oldGrp, check_type_);

  // increase iteration counter after initialization
  ++ls_iters_;

  // -------------------------------------------------
  // update the solution vector and get a trial point
  // -------------------------------------------------
  grp.computeX(oldGrp, dir, step);
  ::NOX::Abstract::Group::ReturnType rtype = ::NOX::Abstract::Group::Ok;
  bool failed = false;
  try
  {
    failed = false;
    rtype = grp.computeF();
    if (rtype != ::NOX::Abstract::Group::Ok) throw_error("compute", "Unable to compute F!");

    /* Safe-guarding of the inner status test:
     * If the outer NormF test is converged for a full step length,
     * we don't have to reduce the step length any further.
     * This additional check becomes necessary, because of cancellation
     * errors and related numerical artifacts. */
    // check the outer status test for the full step length
    outer_tests_ptr_->checkStatus(s, check_type_);

    const NOX::NLN::Solver::LineSearchBased& lsSolver =
        static_cast<const NOX::NLN::Solver::LineSearchBased&>(s);

    const ::NOX::StatusTest::StatusType ostatus = lsSolver.GetStatus<NOX::NLN::StatusTest::NormF>();

    /* Skip the inner status test, if the outer NormF test is
     * already converged! */
    if (ostatus == ::NOX::StatusTest::Converged)
    {
      fp_except_.enable();
      return true;
    }
  }
  // catch error of the computeF method
  catch (const char* e)
  {
    if (not fp_except_.shall_be_caught_) FOUR_C_THROW("An exception occurred: %s", e);

    utils_->out(::NOX::Utils::Warning) << "WARNING: Error caught = " << e << "\n";

    status_ = NOX::NLN::INNER::StatusTest::status_step_too_long;
    failed = true;
  }
  // clear the exception checks after the try/catch block
  fp_except_.clear();

  // -------------------------------------------------
  // print header if desired
  // -------------------------------------------------

  utils_->out(::NOX::Utils::InnerIteration) << "\n"
                                            << ::NOX::Utils::fill(72, '=') << "\n"
                                            << "-- Backtrack Line Search -- \n";

  if (not failed)
  {
    status_ = inner_tests_ptr_->CheckStatus(*this, s, grp, check_type_);
    PrintUpdate(utils_->out(::NOX::Utils::InnerIteration));
  }
  // -------------------------------------------------
  // inner backtracking loop
  // -------------------------------------------------
  while (status_ == NOX::NLN::INNER::StatusTest::status_step_too_long)
  {
    // -------------------------------------------------
    // reduce step length
    // -------------------------------------------------
    prePostOperatorPtr_->run_pre_modify_step_length(s, *this);
    step *= reduction_factor_;

    // -------------------------------------------------
    // - update the solution vector and get a trial point
    // - increase line search step counter
    // -------------------------------------------------
    grp.computeX(oldGrp, dir, step);
    ++ls_iters_;

    try
    {
      rtype = grp.computeF();
      if (rtype != ::NOX::Abstract::Group::Ok) throw_error("compute", "Unable to compute F!");
      status_ = inner_tests_ptr_->CheckStatus(*this, s, grp, check_type_);
      PrintUpdate(utils_->out(::NOX::Utils::InnerIteration));
    }
    // catch error of the computeF method
    catch (const char* e)
    {
      if (not fp_except_.shall_be_caught_) FOUR_C_THROW("An exception occurred: %s", e);

      if (utils_->isPrintType(::NOX::Utils::Warning))
        utils_->out() << "WARNING: Error caught = " << e << "\n";

      status_ = NOX::NLN::INNER::StatusTest::status_step_too_long;
    }

    // clear the exception checks after the try/catch block
    fp_except_.clear();
  }
  // -------------------------------------------------
  // print footer if desired
  // -------------------------------------------------
  utils_->out(::NOX::Utils::InnerIteration) << ::NOX::Utils::fill(72, '=') << "\n";

  if (status_ == NOX::NLN::INNER::StatusTest::status_step_too_short)
    throw_error("compute()",
        "The current step is too short and no "
        "restoration phase is implemented!");
  else if (status_ == NOX::NLN::INNER::StatusTest::status_no_descent_direction)
    throw_error("compute()", "The given search direction is no descent direction!");

  fp_except_.enable();
  return (status_ == NOX::NLN::INNER::StatusTest::status_converged ? true : false);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int NOX::NLN::LineSearch::Backtrack::GetNumIterations() const { return ls_iters_; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const ::NOX::MeritFunction::Generic& NOX::NLN::LineSearch::Backtrack::GetMeritFunction() const
{
  if (merit_function_ptr_.is_null())
    throw_error("GetMeritFunction", "The merit function pointer is not initialized!");

  return *merit_function_ptr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const ::NOX::Abstract::Vector& NOX::NLN::LineSearch::Backtrack::GetSearchDirection() const
{
  if (search_direction_ptr_.is_null())
    throw_error("GetSearchDirection", "The search direction ptr is not initialized!");

  return *search_direction_ptr_;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double NOX::NLN::LineSearch::Backtrack::GetStepLength() const
{
  if (step_ptr_ == nullptr) throw_error("GetStepLength", "Step pointer is nullptr!");

  return *step_ptr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LineSearch::Backtrack::SetStepLength(double step)
{
  if (step_ptr_ == nullptr) throw_error("SetStepLength", "Step pointer is nullptr!");

  *step_ptr_ = step;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::NLN::INNER::StatusTest::StatusType NOX::NLN::LineSearch::Backtrack::CheckInnerStatus(
    const ::NOX::Solver::Generic& solver, const ::NOX::Abstract::Group& grp,
    ::NOX::StatusTest::CheckType checkType) const
{
  return inner_tests_ptr_->CheckStatus(*this, solver, grp, checkType);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LineSearch::Backtrack::PrintUpdate(std::ostream& os) const
{
  // Print the status test parameters at each iteration if requested
  if (status_ == NOX::NLN::INNER::StatusTest::status_step_too_long)
  {
    os << ::NOX::Utils::fill(72, '-') << "\n";
    os << "-- Inner Status Test Results --\n";
    inner_tests_ptr_->Print(os);
    os << ::NOX::Utils::fill(72, '-') << "\n";
  }
  // Print the final parameter values of the status test
  if (status_ != NOX::NLN::INNER::StatusTest::status_step_too_long)
  {
    os << ::NOX::Utils::fill(72, '-') << "\n";
    os << "-- Final Inner Status Test Results --\n";
    inner_tests_ptr_->Print(os);
    os << ::NOX::Utils::fill(72, '-') << "\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LineSearch::Backtrack::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  std::ostringstream msg;
  msg << "ERROR - NOX::NLN::LineSearch::Backtrack::" << functionName << " - " << errorMsg
      << std::endl;
  FOUR_C_THROW(msg.str());
}

FOUR_C_NAMESPACE_CLOSE