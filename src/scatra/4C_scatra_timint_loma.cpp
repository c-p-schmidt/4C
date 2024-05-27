/*----------------------------------------------------------------------*/
/*! \file
\brief scatra time integration for loma
\level 2
 *------------------------------------------------------------------------------------------------*/
#include "4C_scatra_timint_loma.hpp"

#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_sutherland.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | constructor                                          rasthofer 12/13 |
 *----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntLoma::ScaTraTimIntLoma(Teuchos::RCP<DRT::Discretization> dis,
    Teuchos::RCP<CORE::LINALG::Solver> solver, Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<Teuchos::ParameterList> sctratimintparams,
    Teuchos::RCP<Teuchos::ParameterList> extraparams, Teuchos::RCP<IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(dis, solver, sctratimintparams, extraparams, output),
      lomaparams_(params),
      initialmass_(0.0),
      thermpressn_(0.0),
      thermpressnp_(0.0),
      thermpressdtn_(0.0),
      thermpressdtnp_(0.0)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
  return;
}


/*----------------------------------------------------------------------*
 | initialize algorithm                                     rauch 09/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntLoma::Init()
{
  // safety check
  if (CORE::UTILS::IntegralValue<int>(*lomaparams_, "SGS_MATERIAL_UPDATE"))
    FOUR_C_THROW(
        "Material update using subgrid-scale temperature currently not supported for loMa "
        "problems. Read remark in file 'scatra_ele_calc_loma.H'!");

  return;
}


/*----------------------------------------------------------------------*
 | setup algorithm                                          rauch 09/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntLoma::Setup()
{
  SetupSplitter();
  return;
}

/*----------------------------------------------------------------------*
 | setup splitter                                          deanda 11/17 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntLoma::SetupSplitter()
{
  // set up a species-temperature splitter (if more than one scalar)
  if (NumScal() > 1)
  {
    splitter_ = Teuchos::rcp(new CORE::LINALG::MapExtractor);
    CORE::LINALG::CreateMapExtractorFromDiscretization(*discret_, NumScal() - 1, *splitter_);
  }

  return;
}


/*----------------------------------------------------------------------*
 | set initial thermodynamic pressure                          vg 07/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntLoma::set_initial_therm_pressure()
{
  // get thermodynamic pressure from material parameters
  int id = problem_->Materials()->FirstIdByType(CORE::Materials::m_sutherland);
  if (id != -1)  // i.e., Sutherland material found
  {
    const CORE::MAT::PAR::Parameter* mat = problem_->Materials()->ParameterById(id);
    const MAT::PAR::Sutherland* actmat = static_cast<const MAT::PAR::Sutherland*>(mat);

    thermpressn_ = actmat->thermpress_;
  }
  else
  {
    // No Sutherland material found -> now check for temperature-dependent water,
    // which is allowed to be used in TFSI
    int id = problem_->Materials()->FirstIdByType(CORE::Materials::m_tempdepwater);
    if (id != -1)  // i.e., temperature-dependent water found
    {
      // set thermodynamic pressure to zero once and for all
      thermpressn_ = 0.0;
    }
    else
      FOUR_C_THROW(
          "Neiter Sutherland material nor temperature-dependent water found for initial setting of "
          "thermodynamic pressure!");
  }

  // initialize also value at n+1
  // (computed if not constant, otherwise prescribed value remaining)
  thermpressnp_ = thermpressn_;

  // initialize time derivative of thermodynamic pressure at n+1 and n
  // (computed if not constant, otherwise remaining zero)
  thermpressdtnp_ = 0.0;
  thermpressdtn_ = 0.0;

  // compute values at intermediate time steps
  // (only for generalized-alpha time-integration scheme)
  // -> For constant thermodynamic pressure, this is done here once and
  // for all simulation time.
  compute_therm_pressure_intermediate_values();

  return;
}  // SCATRA::ScaTraTimIntLoma::set_initial_therm_pressure


/*----------------------------------------------------------------------*
 | compute initial time derivative of thermodynamic pressure   vg 07/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntLoma::compute_initial_therm_pressure_deriv()
{
  // check for temperature-dependent water, which is allowed to be used in TFSI
  int id = problem_->Materials()->FirstIdByType(CORE::Materials::m_tempdepwater);
  if (id != -1)
    FOUR_C_THROW(
        "Temperature-dependent water found for initial computation of derivative of thermodynamic "
        "pressure -> Set 'CONSTHERMPRES' to 'YES' in FS3I input section!");

  // define element parameter list
  Teuchos::ParameterList eleparams;

  // DO THIS BEFORE PHINP IS SET (ClearState() is called internally!!!!)
  // compute flux approximation and add it to the parameter list
  add_flux_approx_to_parameter_list(eleparams);

  // set scalar vector values needed by elements
  discret_->ClearState();
  discret_->set_state("phinp", phin_);

  // set parameters for element evaluation
  CORE::UTILS::AddEnumClassToParameterList<SCATRA::Action>(
      "action", SCATRA::Action::calc_domain_and_bodyforce, eleparams);

  // the time = 0.0, since this function is called BEFORE the first increment_time_and_step() in
  // InitialCalculations() therefore, the standard set_element_time_parameter() can be used for this
  // method

  // variables for integrals of domain and bodyforce
  Teuchos::RCP<CORE::LINALG::SerialDenseVector> scalars =
      Teuchos::rcp(new CORE::LINALG::SerialDenseVector(2));

  // evaluate domain and bodyforce integral
  discret_->EvaluateScalars(eleparams, scalars);

  // get global integral values
  double pardomint = (*scalars)[0];
  double parbofint = (*scalars)[1];

  // set action for elements
  CORE::UTILS::AddEnumClassToParameterList<SCATRA::BoundaryAction>(
      "action", SCATRA::BoundaryAction::calc_loma_therm_press, eleparams);

  // variables for integrals of normal velocity and diffusive flux
  double normvelint = 0.0;
  double normdifffluxint = 0.0;
  eleparams.set("normal velocity integral", normvelint);
  eleparams.set("normal diffusive flux integral", normdifffluxint);

  // evaluate velocity-divergence and diffusive (minus sign!) flux on boundaries
  // We may use the flux-calculation condition for calculation of fluxes for
  // thermodynamic pressure, since it is usually at the same boundary.
  std::vector<std::string> condnames;
  condnames.push_back("ScaTraFluxCalc");
  for (unsigned int i = 0; i < condnames.size(); i++)
  {
    discret_->evaluate_condition(eleparams, Teuchos::null, Teuchos::null, Teuchos::null,
        Teuchos::null, Teuchos::null, condnames[i]);
  }

  // get integral values on this proc
  normvelint = eleparams.get<double>("normal velocity integral");
  normdifffluxint = eleparams.get<double>("normal diffusive flux integral");

  // get integral values in parallel case
  double parnormvelint = 0.0;
  double parnormdifffluxint = 0.0;
  discret_->Comm().SumAll(&normvelint, &parnormvelint, 1);
  discret_->Comm().SumAll(&normdifffluxint, &parnormdifffluxint, 1);

  // clean up
  discret_->ClearState();

  // compute initial time derivative of thermodynamic pressure
  // (with specific heat ratio fixed to be 1.4)
  const double shr = 1.4;
  thermpressdtn_ =
      (-shr * thermpressn_ * parnormvelint + (shr - 1.0) * (-parnormdifffluxint + parbofint)) /
      pardomint;

  // set time derivative of thermodynamic pressure at n+1 equal to the one at n
  // for following evaluation of intermediate values
  thermpressdtnp_ = thermpressdtn_;

  // compute values at intermediate time steps
  // (only for generalized-alpha time-integration scheme)
  compute_therm_pressure_intermediate_values();

  return;
}  // SCATRA::ScaTraTimIntLoma::compute_initial_therm_pressure_deriv


/*----------------------------------------------------------------------*
 | compute initial total mass in domain                        vg 01/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntLoma::ComputeInitialMass()
{
  // check for temperature-dependent water, which is allowed to be used in TFSI
  int id = problem_->Materials()->FirstIdByType(CORE::Materials::m_tempdepwater);
  if (id != -1)
    FOUR_C_THROW(
        "Temperature-dependent water found for initial computation of mass -> Set 'CONSTHERMPRES' "
        "to 'YES' in FS3I input section!");

  // set scalar values needed by elements
  discret_->ClearState();
  discret_->set_state("phinp", phin_);
  // set action for elements
  Teuchos::ParameterList eleparams;
  CORE::UTILS::AddEnumClassToParameterList<SCATRA::Action>(
      "action", SCATRA::Action::calc_total_and_mean_scalars, eleparams);
  // inverted scalar values are required here
  eleparams.set("inverting", true);
  eleparams.set("calc_grad_phi", false);

  // evaluate integral of inverse temperature
  Teuchos::RCP<CORE::LINALG::SerialDenseVector> scalars =
      Teuchos::rcp(new CORE::LINALG::SerialDenseVector(NumScal() + 1));
  discret_->EvaluateScalars(eleparams, scalars);
  discret_->ClearState();  // clean up

  // compute initial mass times gas constant: R*M_0 = int(1/T_0)*tp
  initialmass_ = (*scalars)[0] * thermpressn_;

  // print out initial total mass
  if (myrank_ == 0)
  {
    std::cout << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "------------+"
              << std::endl;
    std::cout << "Initial total mass in domain (times gas constant): " << initialmass_ << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "------------+"
              << std::endl;
  }

  return;
}  // SCATRA::ScaTraTimIntLoma::ComputeInitialMass


/*----------------------------------------------------------------------*
 | compute thermodynamic pressure from mass conservation       vg 01/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntLoma::compute_therm_pressure_from_mass_cons()
{
  // set scalar values needed by elements
  discret_->ClearState();
  discret_->set_state("phinp", phinp_);
  // set action for elements
  Teuchos::ParameterList eleparams;
  CORE::UTILS::AddEnumClassToParameterList<SCATRA::Action>(
      "action", SCATRA::Action::calc_total_and_mean_scalars, eleparams);
  // inverted scalar values are required here
  eleparams.set("inverting", true);
  eleparams.set("calc_grad_phi", false);

  // evaluate integral of inverse temperature
  Teuchos::RCP<CORE::LINALG::SerialDenseVector> scalars =
      Teuchos::rcp(new CORE::LINALG::SerialDenseVector(NumScal() + 1));
  discret_->EvaluateScalars(eleparams, scalars);
  discret_->ClearState();  // clean up

  // compute thermodynamic pressure: tp = R*M_0/int(1/T)
  thermpressnp_ = initialmass_ / (*scalars)[0];

  // print out thermodynamic pressure
  if (myrank_ == 0)
  {
    std::cout << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "------------+"
              << std::endl;
    std::cout << "Thermodynamic pressure from mass conservation: " << thermpressnp_ << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "------------+"
              << std::endl;
  }

  // compute time derivative of thermodynamic pressure at time step n+1
  compute_therm_pressure_time_derivative();

  // compute values at intermediate time steps
  // (only for generalized-alpha time-integration scheme)
  compute_therm_pressure_intermediate_values();

  return;
}  // SCATRA::ScaTraTimIntLoma::compute_therm_pressure_from_mass_cons


/*----------------------------------------------------------------------*
 | add parameters depending on the problem              rasthofer 12/13 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntLoma::add_problem_specific_parameters_and_vectors(
    Teuchos::ParameterList& params  //!< parameter list
)
{
  add_therm_press_to_parameter_list(params);
  return;
}

FOUR_C_NAMESPACE_CLOSE