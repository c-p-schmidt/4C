/*----------------------------------------------------------------------*/
/*! \file
\brief Base class for the inverse analysis

\level 3

*/
/*----------------------------------------------------------------------*/
#include "inv_analysis_base.H"
#include "inv_analysis_matpar_manager.H"
#include "inv_analysis_objective_funct.H"
#include "inv_analysis_regularization_base.H"
#include "inv_analysis_optimizer_base.H"
#include "inv_analysis_initial_guess.H"
#include "inv_analysis_resulttest.H"
#include "lib_discret.H"

// to modify the IO
#include "lib_globalproblem.H"
#include "io_control.H"
#include "io.H"


/*----------------------------------------------------------------------*/
INVANA::InvanaBase::InvanaBase()
    : discret_(Teuchos::null),
      objfunct_(Teuchos::null),
      matman_(Teuchos::null),
      regman_(Teuchos::null),
      fprestart_(0),
      isinit_(false)
{
  ;
}

/*----------------------------------------------------------------------*/
void INVANA::InvanaBase::Init(Teuchos::RCP<DRT::Discretization> discret,
    Teuchos::RCP<INVANA::ObjectiveFunct> objfunct, Teuchos::RCP<INVANA::MatParManager> matman,
    Teuchos::RCP<INVANA::RegularizationBase> regman, Teuchos::RCP<INVANA::InitialGuess> initguess)
{
  discret_ = discret;
  objfunct_ = objfunct;
  matman_ = matman;
  regman_ = regman;
  initguess_ = initguess;

  SetupIO();

  isinit_ = true;
}

/*----------------------------------------------------------------------*/
const Epetra_Comm& INVANA::InvanaBase::Comm() { return discret_->Comm(); }

/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> INVANA::InvanaBase::VectorRowLayout()
{
  return matman_->ParamLayoutMapUnique();
}

/*----------------------------------------------------------------------*/
void INVANA::InvanaBase::SetupIO()
{
  // a problem pointer for convenience
  DRT::Problem* problem = DRT::Problem::Instance();

  // binio from the binio section (applied to the forward problem output only)
  bool binio = DRT::INPUT::IntegralValue<int>(DRT::Problem::Instance()->IOParams(), "OUTPUT_BIN");

  // read forward problem restart stuff
  const Teuchos::ParameterList& invp = DRT::Problem::Instance()->StatInverseAnalysisParams();
  fprestart_ = invp.get<int>("FPRESTART");
  std::string restartfilename = invp.get<std::string>("FPOUTPUTFILENAME");

  // modify filename for the forward problem to ..._forward
  std::string filename = problem->OutputControlFile()->FileName();
  std::string prefix = problem->OutputControlFile()->FileNameOnlyPrefix();
  size_t pos = filename.rfind('/');
  size_t pos2 = prefix.rfind('-');
  filename = filename.substr(0, pos + 1) + prefix.substr(0, pos2) + "_forward" +
             filename.substr(pos + 1 + prefix.length());

  // a new controlfile with this name
  Teuchos::RCP<IO::OutputControl> controlfile = Teuchos::null;
  controlfile = Teuchos::rcp(new IO::OutputControl(Discret()->Comm(), problem->ProblemName(),
      problem->SpatialApproximationType(), problem->OutputControlFile()->InputFileName(), filename,
      problem->NDim(), 0, problem->OutputControlFile()->FileSteps(), binio));

  // give this one to the discretization to be used in the field output
  Discret()->Writer()->SetOutput(controlfile);

  // set forward problem restart to the restartname in case
  if (restartfilename.compare("none") != 0)
  {
    inputcontrol_ = Teuchos::rcp(new IO::InputControl(restartfilename, Discret()->Comm()));
  }

  // sanity check
  if ((restartfilename.compare("none") == 0) and (fprestart_ != 0))
    dserror(
        "Set a proper forward problem restart name to be able to restart a forward simulation!");

  return;
}

/*----------------------------------------------------------------------*/
Teuchos::RCP<IO::InputControl>& INVANA::InvanaBase::InputControl()
{
  if (inputcontrol_ == Teuchos::null)
    dserror("Don't ask for the forward problem input control. It wasn't set!");

  return inputcontrol_;
}