/*---------------------------------------------------------------------*/
/*! \file

\brief Incomplete! - Purpose: Templated Evaluate file for blood scatra line3 element containing the
action types for a reduced blood scatra element RedAirBloodScatraLine3. The actual implementation of
the routines called during the possible actions is contained in red_air_blood_scatraLine3_impl.cpp


\level 3

*/
/*---------------------------------------------------------------------*/


#include "4C_discretization_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_list.hpp"
#include "4C_red_airways_air_blood_scatraLine3_impl.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN



/*---------------------------------------------------------------------*
 |evaluate the element (public)                            ismail 09/12|
 *---------------------------------------------------------------------*/
int DRT::ELEMENTS::RedAirBloodScatraLine3::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseMatrix& elemat1, CORE::LINALG::SerialDenseMatrix& elemat2,
    CORE::LINALG::SerialDenseVector& elevec1, CORE::LINALG::SerialDenseVector& elevec2,
    CORE::LINALG::SerialDenseVector& elevec3)
{
  DRT::ELEMENTS::RedAirBloodScatraLine3::ActionType act = RedAirBloodScatraLine3::none;

  // get the action required
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "calc_sys_matrix_rhs")
    act = RedAirBloodScatraLine3::calc_sys_matrix_rhs;
  else if (action == "calc_sys_matrix_rhs_iad")
    act = RedAirBloodScatraLine3::calc_sys_matrix_rhs_iad;
  else if (action == "get_initial_state")
    act = RedAirBloodScatraLine3::get_initial_state;
  else if (action == "set_bc")
    act = RedAirBloodScatraLine3::set_bc;
  else if (action == "calc_flow_rates")
    act = RedAirBloodScatraLine3::calc_flow_rates;
  else if (action == "calc_elem_volumes")
    act = RedAirBloodScatraLine3::calc_elem_volumes;
  else if (action == "get_coupled_values")
    act = RedAirBloodScatraLine3::get_coupled_values;
  else if (action == "get_junction_volume_mix")
    act = RedAirBloodScatraLine3::get_junction_volume_mix;
  else if (action == "solve_scatra")
    act = RedAirBloodScatraLine3::solve_scatra;
  else if (action == "calc_cfl")
    act = RedAirBloodScatraLine3::calc_cfl;
  else if (action == "solve_junction_scatra")
    act = RedAirBloodScatraLine3::solve_junction_scatra;
  else if (action == "eval_nodal_essential_values")
    act = RedAirBloodScatraLine3::eval_nodal_ess_vals;
  else if (action == "solve_blood_air_transport")
    act = RedAirBloodScatraLine3::solve_blood_air_transport;
  else if (action == "update_scatra")
    act = RedAirBloodScatraLine3::update_scatra;
  else if (action == "update_elem12_scatra")
    act = RedAirBloodScatraLine3::update_elem12_scatra;
  else if (action == "eval_PO2_from_concentration")
    act = RedAirBloodScatraLine3::eval_PO2_from_concentration;
  else
  {
    char errorout[200];
    sprintf(errorout, "Unknown type of action (%s) for reduced dimensional acinus", action.c_str());

    FOUR_C_THROW(errorout);
  }

  /*
  Here must add the steps for evaluating an element
  */
  Teuchos::RCP<CORE::MAT::Material> mat = Material();

  switch (act)
  {
    case calc_sys_matrix_rhs:
    {
    }
    break;
    case calc_sys_matrix_rhs_iad:
    {
    }
    break;
    case get_initial_state:
    {
    }
    break;
    case set_bc:
    {
    }
    break;
    case calc_flow_rates:
    {
    }
    break;
    case calc_elem_volumes:
    {
    }
    break;
    case get_coupled_values:
    {
    }
    break;
    case get_junction_volume_mix:
    {
      // do nothing
    }
    break;
    case solve_scatra:
    {
      // do nothing
    }
    break;
    case calc_cfl:
    {
      // do nothing
    }
    break;
    case solve_junction_scatra:
    {
      // do nothing
    }
    break;
    case update_scatra:
    {
      // do nothing
    }
    break;
    case update_elem12_scatra:
    {
      // do nothing
    }
    break;
    case eval_nodal_ess_vals:
    {
      // do nothing
    }
    break;
    case eval_PO2_from_concentration:
    {
      // do nothing
    }
    break;
    case solve_blood_air_transport:
    {
      DRT::ELEMENTS::RedAirBloodScatraLine3ImplInterface::Impl(this)->solve_blood_air_transport(
          this, elevec1, elevec2, params, discretization, lm, mat);
    }
    break;
    default:
      FOUR_C_THROW("Unkown type of action for reduced dimensional acinuss");
  }  // end of switch(act)

  return 0;
}  // end of DRT::ELEMENTS::RedAirBloodScatraLine3::Evaluate


int DRT::ELEMENTS::RedAirBloodScatraLine3::evaluate_neumann(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, CORE::Conditions::Condition& condition,
    std::vector<int>& lm, CORE::LINALG::SerialDenseVector& elevec1,
    CORE::LINALG::SerialDenseMatrix* elemat1)
{
  return 0;
}

/*----------------------------------------------------------------------*
 |  do nothing (public)                                     ismail 09/12|
 |                                                                      |
 |  The function is just a dummy.                                       |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::RedAirBloodScatraLine3::EvaluateDirichlet(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, CORE::Conditions::Condition& condition,
    std::vector<int>& lm, CORE::LINALG::SerialDenseVector& elevec1)
{
  return 0;
}


// get optimal gaussrule for discretization type
CORE::FE::GaussRule1D DRT::ELEMENTS::RedAirBloodScatraLine3::get_optimal_gaussrule(
    const CORE::FE::CellType& distype)
{
  CORE::FE::GaussRule1D rule = CORE::FE::GaussRule1D::undefined;
  switch (distype)
  {
    case CORE::FE::CellType::line2:
      rule = CORE::FE::GaussRule1D::line_2point;
      break;
    case CORE::FE::CellType::line3:
      rule = CORE::FE::GaussRule1D::line_3point;
      break;
    default:
      FOUR_C_THROW("unknown number of nodes for gaussrule initialization");
  }
  return rule;
}


// check, whether higher order derivatives for shape functions (dxdx, dxdy, ...) are necessary
bool DRT::ELEMENTS::RedAirBloodScatraLine3::is_higher_order_element(
    const CORE::FE::CellType distype) const
{
  bool hoel = true;
  switch (distype)
  {
    case CORE::FE::CellType::line3:
      hoel = true;
      break;
    case CORE::FE::CellType::line2:
      hoel = false;
      break;
    default:
      FOUR_C_THROW("distype unknown!");
  }
  return hoel;
}

FOUR_C_NAMESPACE_CLOSE