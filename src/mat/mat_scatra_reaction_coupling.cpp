/*----------------------------------------------------------------------*/
/*! \file
 \brief helper class encapsulating the reaction terms and its derivatives

   \level 3

 *----------------------------------------------------------------------*/

#include "mat_scatra_reaction_coupling.H"

#include "lib_globalproblem.H"
#include "lib_get_functionofanything.H"

/*----------------------------------------------------------------------*
 * factory method                                            vuong 09/16
 *----------------------------------------------------------------------*/
Teuchos::RCP<MAT::PAR::REACTIONCOUPLING::ReactionInterface>
MAT::PAR::REACTIONCOUPLING::ReactionInterface::CreateReaction(
    MAT::PAR::reaction_coupling couplingtype,  //!< coupling type defining reaction
    bool isreacstart,                          //!< flag for reaction start feature
    const std::vector<double>& reacstart       //!< reaction start vector
)
{
  Teuchos::RCP<MAT::PAR::REACTIONCOUPLING::ReactionInterface> tmpreaction = Teuchos::null;

  switch (couplingtype)
  {
    case MAT::PAR::reac_coup_simple_multiplicative:  // reaction of type A*B*C:
    {
      tmpreaction = Teuchos::rcp(new MAT::PAR::REACTIONCOUPLING::SimpleMultiplicative());
      break;
    }
    case MAT::PAR::reac_coup_power_multiplicative:  // reaction of type A^2*B^-1.5*C:
    {
      tmpreaction = Teuchos::rcp(new MAT::PAR::REACTIONCOUPLING::PowerMultiplicative());
      break;
    }
    case MAT::PAR::reac_coup_constant:  // constant source term:
    {
      tmpreaction = Teuchos::rcp(new MAT::PAR::REACTIONCOUPLING::Constant());
      break;
    }
    case MAT::PAR::reac_coup_michaelis_menten:  // reaction of type A*B/(B+4)
    {
      tmpreaction = Teuchos::rcp(new MAT::PAR::REACTIONCOUPLING::MichaelisMenten());
      break;
    }
    case MAT::PAR::reac_coup_byfunction:  // reaction by function
    {
      tmpreaction = Teuchos::rcp(new MAT::PAR::REACTIONCOUPLING::ByFunction());
      break;
    }
    case MAT::PAR::reac_coup_none:
      dserror("reac_coup_none is not a valid coupling");
      break;
    default:
      dserror("The couplingtype %i is not a valid coupling type.", couplingtype);
      break;
  }

  // we always use potentially scaled phis for the reactions (for reference concentrations)
  Teuchos::RCP<MAT::PAR::REACTIONCOUPLING::ReactionInterface> scaledreaction =
      Teuchos::rcp(new MAT::PAR::REACTIONCOUPLING::ReactionWithPhiScaling(tmpreaction));

  Teuchos::RCP<MAT::PAR::REACTIONCOUPLING::ReactionInterface> reaction = Teuchos::null;
  // in case of reac start feature, wrap it one more time
  if (isreacstart)
    reaction = Teuchos::rcp(new MAT::PAR::REACTIONCOUPLING::ReacStart(scaledreaction, reacstart));
  else
    reaction = scaledreaction;

  return reaction;
}


/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  access method for calculating advanced reaction terms    vuong 09/16 |
 *----------------------------------------------------------------------*/
double MAT::PAR::REACTIONCOUPLING::ReactionWithPhiScaling::CalcReaBodyForceTerm(
    const int k,                       //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
)
{
  // modify the phinp vector if neccessary (e.g. for reference concentrations)
  std::vector<double> phinp_mod(ModifyPhi(phinp, scale_phi));

  // call the real evaluation
  return reaction_->CalcReaBodyForceTerm(
      k, numscal, phinp_mod, constants, couprole, scale_reac, scale_phi);
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::ReactionWithPhiScaling::CalcReaBodyForceDeriv(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
)
{
  // modify the phinp vector if necessary (e.g. for reference concentrations)
  std::vector<double> phinp_mod(ModifyPhi(phinp, scale_phi));

  // call the real evaluation
  reaction_->CalcReaBodyForceDeriv(
      k, numscal, derivs, phinp_mod, constants, couprole, scale_reac, scale_phi);

  return;
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives after additional    |
 |  variables (e.g. for monolithic coupling)                     kremheller 07/17 |
 *--------------------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::ReactionWithPhiScaling::CalcReaBodyForceDerivAddVariables(
    const int k,                  //!< current scalar id
    std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<std::pair<std::string, double>>&
        constants,                        //!< constants (including scalar values phinp)
    const std::vector<double>& couprole,  //!< coupling role vector
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
)
{
  // modify the phinp vector if necessary (e.g. for reference concentrations)
  // std::vector<double> phinp_mod(ModifyPhi(phinp,scale_phi));
  if (fabs(scale_phi - 1.0) > 1.0e-14)
  {
    dserror("scale_phi is not equal to 1.0, you should make your own modify phi function");
  }

  // call the real evaluation
  reaction_->CalcReaBodyForceDerivAddVariables(
      k, derivs, variables, constants, couprole, scale_reac, scale_phi);

  return;
}

/*----------------------------------------------------------------------------------*
 |  Modify concentrations according to scaling                         vuong 09/16 |
 *----------------------------------------------------------------------------------*/
std::vector<double> MAT::PAR::REACTIONCOUPLING::ReactionWithPhiScaling::ModifyPhi(
    const std::vector<double>& phinp, double scale_phi)
{
  // copy the vector
  std::vector<double> phinp_mod(phinp);

  if (scale_phi != 1.0)
    // scale the vector and save the result in phinp_mod
    std::transform(phinp.begin(), phinp.end(), phinp_mod.begin(),
        std::bind1st(std::multiplies<double>(), scale_phi));

  return phinp_mod;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::ReacStart::Initialize(int numscal,  //!< number of scalars
    const std::vector<double>& couprole                              //!< coupling role vector
)
{
  reaction_->Initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 |  access method for calculating advanced reaction terms    vuong 09/16 |
 *----------------------------------------------------------------------*/
double MAT::PAR::REACTIONCOUPLING::ReacStart::CalcReaBodyForceTerm(
    const int k,                       //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
)
{
  // modify the phinp vector for reaction start feature
  std::vector<double> phinp_mod(ModifyPhi(phinp));

  // call the real evaluation
  return reaction_->CalcReaBodyForceTerm(
      k, numscal, phinp_mod, constants, couprole, scale_reac, scale_phi);
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::ReacStart::CalcReaBodyForceDeriv(int k,  //!< current scalar id
    int numscal,                                                          //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
)
{
  // modify the phinp vector for reaction start feature
  std::vector<double> phinp_mod(ModifyPhi(phinp));

  // temporary vector of same size as derivs
  std::vector<double> myderivs(derivs.size(), 0.0);

  // call reaction evaluation
  reaction_->CalcReaBodyForceDeriv(
      k, numscal, myderivs, phinp_mod, constants, couprole, scale_reac, scale_phi);

  for (int toderive = 0; toderive < numscal; toderive++)
  {
    // only copy the value if reaction has already started
    if (not(reacstart_[toderive] > 0 and phinp_mod[toderive] == 0.0))
      derivs[toderive] += myderivs[toderive];
  }

  return;
}

/*----------------------------------------------------------------------------------*
 |  Modify concentrations according to reacstart vector                  vuong 09/16 |
 *----------------------------------------------------------------------------------*/
std::vector<double> MAT::PAR::REACTIONCOUPLING::ReacStart::ModifyPhi(
    const std::vector<double>& phinp)
{
  // copy the vector
  std::vector<double> phinp_mod(phinp);

  for (unsigned int ii = 0; ii < phinp_mod.size(); ii++)
  {
    phinp_mod[ii] -= reacstart_[ii];
    if (phinp_mod[ii] < 0.0) phinp_mod[ii] = 0.0;
  }

  return phinp_mod;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::SimpleMultiplicative::Initialize(
    int numscal,                         //!< number of scalars
    const std::vector<double>& couprole  //!< coupling role vector
)
{
  ReactionBase::Initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 |  helper for calculating advanced reaction terms           vuong 09/16 |
 *----------------------------------------------------------------------*/
double MAT::PAR::REACTIONCOUPLING::SimpleMultiplicative::CalcReaBodyForceTerm(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  double bftfac = 1.0;

  for (int ii = 0; ii < numscal; ii++)
  {
    if (couprole[ii] != 0)
    {
      bftfac *= phinp[ii];
    }
  }

  return scale_reac * bftfac;
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::SimpleMultiplicative::CalcReaBodyForceDeriv(
    const int k,                       //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    const double
        scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  for (int toderive = 0; toderive < numscal; toderive++)
  {
    double bfdmfac = 1.0;
    if (couprole[toderive] != 0)
    {
      for (int ii = 0; ii < numscal; ii++)
      {
        if (couprole[ii] != 0 and ii != toderive) bfdmfac *= phinp[ii];
      }
    }
    else
      bfdmfac = 0.0;

    derivs[toderive] += scale_reac * bfdmfac;
  }

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::PowerMultiplicative::Initialize(
    int numscal,                         //!< number of scalars
    const std::vector<double>& couprole  //!< coupling role vector
)
{
  ReactionBase::Initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 |  helper for calculating advanced reaction terms           vuong 09/16 |
 *----------------------------------------------------------------------*/
double MAT::PAR::REACTIONCOUPLING::PowerMultiplicative::CalcReaBodyForceTerm(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  double bftfac = 1.0;

  for (int ii = 0; ii < numscal; ii++)
  {
    if (couprole[ii] != 0)
    {
      bftfac *= std::pow(phinp[ii], couprole[ii]);
    }
  }

  return scale_reac * bftfac;
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::PowerMultiplicative::CalcReaBodyForceDeriv(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  for (int toderive = 0; toderive < numscal; toderive++)
  {
    double bfdmfac = 1.0;
    if (couprole[toderive] != 0)
    {
      for (int ii = 0; ii < numscal; ii++)
      {
        if (couprole[ii] != 0 and ii != toderive)
          bfdmfac *= std::pow(phinp[ii], couprole[ii]);
        else if (couprole[ii] != 0 and ii == toderive)
          bfdmfac *= couprole[ii] * std::pow(phinp[ii], couprole[ii] - 1.0);
      }
    }
    else
      bfdmfac = 0.0;

    derivs[toderive] += scale_reac * bfdmfac;
  }

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::Constant::Initialize(int numscal,  //!< number of scalars
    const std::vector<double>& couprole                             //!< coupling role vector
)
{
  ReactionBase::Initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 |  helper for calculating advanced reaction terms           vuong 09/16 |
 *----------------------------------------------------------------------*/
double MAT::PAR::REACTIONCOUPLING::Constant::CalcReaBodyForceTerm(int k,  //!< current scalar id
    int numscal,                                                          //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  return scale_reac;
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::Constant::CalcReaBodyForceDeriv(int k,  //!< current scalar id
    int numscal,                                                         //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  // zero derivative -> do nothing
  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::MichaelisMenten::Initialize(int numscal,  //!< number of scalars
    const std::vector<double>& couprole                                    //!< coupling role vector
)
{
  ReactionBase::Initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 |  helper for calculating advanced reaction terms           vuong 09/16 |
 *----------------------------------------------------------------------*/
double MAT::PAR::REACTIONCOUPLING::MichaelisMenten::CalcReaBodyForceTerm(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  double bftfac = 1.0;

  for (int ii = 0; ii < numscal; ii++)
  {
    if (couprole[ii] > 0.0)  // and (ii!=k))
      bftfac *= phinp[ii] / (couprole[ii] + phinp[ii]);
    else if (couprole[ii] < 0.0)  // and (ii!=k))
      bftfac *= phinp[ii];
  }

  return scale_reac * bftfac;
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::MichaelisMenten::CalcReaBodyForceDeriv(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  for (int toderive = 0; toderive < numscal; toderive++)
  {
    double bfdmfac = 1.0;
    for (int ii = 0; ii < numscal; ii++)
    {
      if (ii != toderive)
      {
        if (couprole[ii] > 0.0)
          bfdmfac *= phinp[ii] / (couprole[ii] + phinp[ii]);
        else if (couprole[ii] < 0.0)
          bfdmfac *= phinp[ii];
        else
          bfdmfac *= 1;
      }
      else
      {
        if (couprole[ii] > 0.0)
          bfdmfac *= couprole[ii] / (std::pow((phinp[ii] + couprole[ii]), 2));
        else if (couprole[ii] < 0.0)
          bfdmfac *= 1;
        else
          bfdmfac = 0;
      }
    }
    derivs[toderive] += scale_reac * bfdmfac;
  }

  return;
}


/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::ByFunction::Initialize(int numscal,  //!< number of scalars
    const std::vector<double>& couprole                               //!< coupling role vector
)
{
  switch (DRT::Problem::Instance()->NDim())
  {
    case 1:
      return InitializeInternal<1>(numscal, couprole);
    case 2:
      return InitializeInternal<2>(numscal, couprole);

    case 3:
      return InitializeInternal<3>(numscal, couprole);

    default:
      dserror("Unsupported dimension %d.", DRT::Problem::Instance()->NDim());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void MAT::PAR::REACTIONCOUPLING::ByFunction::InitializeInternal(int numscal,  //!< number of scalars
    const std::vector<double>& couprole  //!< coupling role vector
)
{
  if (not IsInit())
  {
    variables_.reserve(numscal);
    for (int ii = 0; ii < numscal; ii++)
    {
      // we take the value in couprole list as function ID
      const int functID = round(couprole[ii]);
      if (functID != 0)
      {
        if (DRT::UTILS::GetFunctionOfAnything(functID - 1).NumberComponents() != 1)
          dserror("expected only one component for the reaction evaluation");

        for (int k = 0; k < numscal; k++)
        {
          // construct the strings for scalar
          std::ostringstream temp;
          temp << k + 1;
          std::string name = "phi" + temp.str();


          // save the phi values with correct name
          variables_.emplace_back(name, 0.0);
        }
      }
    }
  }


  // call base class
  ReactionBase::Initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double MAT::PAR::REACTIONCOUPLING::ByFunction::CalcReaBodyForceTerm(
    const int k,                       //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                       //!< stoichometry)
)
{
  switch (DRT::Problem::Instance()->NDim())
  {
    case 1:
      return CalcReaBodyForceTermInternal<1>(k, numscal, phinp, constants, couprole, scale_reac);
    case 2:
      return CalcReaBodyForceTermInternal<2>(k, numscal, phinp, constants, couprole, scale_reac);

    case 3:
      return CalcReaBodyForceTermInternal<3>(k, numscal, phinp, constants, couprole, scale_reac);

    default:
      dserror("Unsupported dimension %d.", DRT::Problem::Instance()->NDim());
      return 0.0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
double MAT::PAR::REACTIONCOUPLING::ByFunction::CalcReaBodyForceTermInternal(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  // copy phi vector in different format to be read by the function
  BuildPhiVectorForFunction(phinp, numscal);

  std::vector<std::pair<std::string, double>> variables_for_parser_evaluation = variables_;

  // add possible time dependency

  variables_for_parser_evaluation.emplace_back("t", 0.0);

  // add possible spatial dependency

  variables_for_parser_evaluation.emplace_back("x", 0.0);
  variables_for_parser_evaluation.emplace_back("y", 0.0);
  variables_for_parser_evaluation.emplace_back("z", 0.0);

  // evaluate reaction term
  double bftfac = DRT::UTILS::GetFunctionOfAnything(round(couprole[k]) - 1)
                      .Evaluate(variables_for_parser_evaluation, constants, 0);

  return scale_reac * bftfac;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::ByFunction::CalcReaBodyForceDeriv(int k,  //!< current scalar id
    int numscal,                                                           //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  switch (DRT::Problem::Instance()->NDim())
  {
    case 1:
      return CalcReaBodyForceDerivInternal<1>(
          k, numscal, derivs, phinp, constants, couprole, scale_reac);
    case 2:
      return CalcReaBodyForceDerivInternal<2>(
          k, numscal, derivs, phinp, constants, couprole, scale_reac);
      ;
    case 3:
      return CalcReaBodyForceDerivInternal<3>(
          k, numscal, derivs, phinp, constants, couprole, scale_reac);
      ;
    default:
      dserror("Unsupported dimension %d.", DRT::Problem::Instance()->NDim());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void MAT::PAR::REACTIONCOUPLING::ByFunction::CalcReaBodyForceDerivInternal(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  // copy phi vector in different format to be read by the function
  BuildPhiVectorForFunction(phinp, numscal);

  std::vector<std::pair<std::string, double>> constants_for_parser_evaluation = constants;

  // add possible time dependency

  constants_for_parser_evaluation.emplace_back("t", 0.0);

  // add possible spatial dependency

  constants_for_parser_evaluation.emplace_back("x", 0.0);
  constants_for_parser_evaluation.emplace_back("y", 0.0);
  constants_for_parser_evaluation.emplace_back("z", 0.0);


  // evaluate the derivatives of the reaction term
  std::vector<double> myderivs =
      DRT::UTILS::GetFunctionOfAnything(round(couprole[k]) - 1)
          .EvaluateDerivative(variables_, constants_for_parser_evaluation, 0);

  // add it to derivs
  for (int toderive = 0; toderive < numscal; toderive++)
    derivs[toderive] += scale_reac * myderivs[toderive];

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::ByFunction::CalcReaBodyForceDerivAddVariables(
    const int k,                  //!< current scalar id
    std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<std::pair<std::string, double>>&
        constants,                        //!< constants (including scalar values phinp)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  switch (DRT::Problem::Instance()->NDim())
  {
    case 1:
      return CalcReaBodyForceDerivAddVariablesInternal<1>(
          k, derivs, variables, constants, couprole, scale_reac);
    case 2:
      return CalcReaBodyForceDerivAddVariablesInternal<2>(
          k, derivs, variables, constants, couprole, scale_reac);
      ;
    case 3:
      return CalcReaBodyForceDerivAddVariablesInternal<3>(
          k, derivs, variables, constants, couprole, scale_reac);
      ;
    default:
      dserror("Unsupported dimension %d.", DRT::Problem::Instance()->NDim());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void MAT::PAR::REACTIONCOUPLING::ByFunction::CalcReaBodyForceDerivAddVariablesInternal(
    const int k,                  //!< current scalar id
    std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<std::pair<std::string, double>>&
        constants,                        //!< constants (including scalar values phinp)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  // evaluate the derivatives of the reaction term
  std::vector<double> myderivs = DRT::UTILS::GetFunctionOfAnything(round(couprole[k]) - 1)
                                     .EvaluateDerivative(variables, constants, 0);

  if (myderivs.size() != derivs.size())
  {
    dserror("mismatch in dimensions, Input %d, Output %d", derivs.size(), myderivs.size());
  }

  // add it to derivs
  for (unsigned toderive = 0; toderive < variables.size(); toderive++)
    derivs[toderive] += scale_reac * myderivs[toderive];

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MAT::PAR::REACTIONCOUPLING::ByFunction::AddAdditionalVariables(
    const int k,                                                   //!< current scalar id
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<double>& couprole                            //!< coupling role vector
)
{
  switch (DRT::Problem::Instance()->NDim())
  {
    case 1:
      return AddAdditionalVariablesInternal<1>(k, variables, couprole);
    case 2:
      return AddAdditionalVariablesInternal<2>(k, variables, couprole);
      ;
    case 3:
      return AddAdditionalVariablesInternal<3>(k, variables, couprole);
      ;
    default:
      dserror("Unsupported dimension %d.", DRT::Problem::Instance()->NDim());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void MAT::PAR::REACTIONCOUPLING::ByFunction::AddAdditionalVariablesInternal(
    const int k,                                                   //!< current scalar id
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<double>& couprole                            //!< coupling role vector
)
{
  // nothing to do

  return;
}

/*---------------------------------------------------------------------------------/
 | helper for evaluation by function                                     vuong 08/16 |
/--------------------------------------------------------------------------------- */
void MAT::PAR::REACTIONCOUPLING::ByFunction::BuildPhiVectorForFunction(
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    int numscal                        //!< number of scalars
)
{
  // note: we use the fact that the 'variables_' vector is ordered in the same way
  //       as the phi vector!
  for (int ii = 0; ii < numscal; ii++)
  {
    variables_[ii].second = phinp[ii];
  }
  return;
}