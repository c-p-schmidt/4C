// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_list.hpp"
#include "4C_so3_poro_scatra.hpp"
#include "4C_so3_poro_scatra_eletypes.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3PoroScatra<So3Ele, distype>::pre_evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  if (la.size() > 2)
  {
    if (discretization.has_state(2, "scalar"))
    {
      // check if you can get the scalar state
      std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp =
          discretization.get_state(2, "scalar");

      // extract local values of the global vectors
      std::vector<double> myscalar = Core::FE::extract_values(*scalarnp, la[2].lm_);

      if (So3Ele::num_material() < 2)
        FOUR_C_THROW("no second material defined for Wall poro element!");
      std::shared_ptr<Core::Mat::Material> scatramat = So3Ele::material(2);

      int numscal = 1;
      if (scatramat->material_type() == Core::Materials::m_matlist or
          scatramat->material_type() == Core::Materials::m_matlist_reactions)
      {
        std::shared_ptr<Mat::MatList> matlist = std::dynamic_pointer_cast<Mat::MatList>(scatramat);
        numscal = matlist->num_mat();
      }

      std::shared_ptr<std::vector<double>> scalar =
          std::make_shared<std::vector<double>>(numscal, 0.0);
      if ((int)myscalar.size() != numscal * numnod_) FOUR_C_THROW("sizes do not match!");

      for (int i = 0; i < numnod_; i++)
        for (int j = 0; j < numscal; j++) scalar->at(j) += myscalar[numscal * i + j] / numnod_;

      params.set("scalar", scalar);
    }
  }
  else
  {
    const double time = params.get("total time", 0.0);
    // find out whether we will use a time curve and get the factor
    int num = 1;  // TO BE READ FROM INPUTFILE AT EACH ELEMENT!!!
    std::vector<double> xrefe;
    xrefe.resize(3);
    Core::Nodes::Node** nodes = my::nodes();
    // get displacements of this element
    //  Core::FE::extract_my_values(*disp,mydisp,lm);
    for (int i = 0; i < numnod_; ++i)
    {
      const auto& x = nodes[i]->x();
      xrefe[0] += x[0] / numnod_;
      xrefe[1] += x[1] / numnod_;
      xrefe[2] += x[2] / numnod_;
    }
    const double* coordgpref = xrefe.data();
    double functfac =
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(num).evaluate(
            coordgpref, time, 0);
    params.set<double>("scalar", functfac);
  }
}

template <class So3Ele, Core::FE::CellType distype>
int Discret::Elements::So3PoroScatra<So3Ele, distype>::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  if (!my::init_) FOUR_C_THROW("internal element data not initialized!");

  // set the pointer to the parameter list in element
  So3Ele::set_params_interface_ptr(params);

  // start with "none"
  Core::Elements::ActionType act = Core::Elements::none;

  if (So3Ele::is_params_interface())
  {
    act = So3Ele::params_interface().get_action_type();
  }
  else
  {
    // get the required action
    std::string action = params.get<std::string>("action", "none");
    if (action == "none")
      FOUR_C_THROW("No action supplied");
    else if (action == "struct_poro_calc_scatracoupling")
      act = Core::Elements::struct_poro_calc_scatracoupling;
  }

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // off diagonal terms in stiffness matrix for monolithic coupling
    case Core::Elements::struct_poro_calc_scatracoupling:
      // no coupling-> return
      break;
    //==================================================================================
    default:
    {
      my::evaluate(params, discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra,
          elevec2_epetra, elevec3_epetra);
    }
    break;
  }

  return 0;
}

FOUR_C_NAMESPACE_CLOSE

#include "4C_so3_poro_scatra.inst.hpp"
