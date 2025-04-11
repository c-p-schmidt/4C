// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MYOCARD_TENTUSSCHER_HPP
#define FOUR_C_MAT_MYOCARD_TENTUSSCHER_HPP

/*----------------------------------------------------------------------*
 |  headers                                                  cbert 08/13 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_myocard_general.hpp"
#include "4C_mat_myocard_tools.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/// Myocard material according to [1]
///
/// This is a reaction-diffusion law of anisotropic, instationary electric conductivity in cardiac
/// muscle tissue
///
/// <h3>References</h3>
/// <ul>
/// <li> [1] KH ten Tusscher et. al., "Alternans and spiral breakup in a human ventricular tissue
/// model", Am J Physiol Heart Circ Physiol 291 (2006) 1088-1100
/// </ul>
///



class MyocardTenTusscher : public MyocardGeneral

{
 public:
  /// construct empty material object
  MyocardTenTusscher();

  /// construct empty material object
  explicit MyocardTenTusscher(const double eps_deriv_myocard, const std::string& tissue);

  /// compute reaction coefficient
  double rea_coeff(const double phi, const double dt) override;

  ///  returns number of internal state variables of the material
  int get_number_of_internal_state_variables() const override;

  ///  return current internal state of the material
  double get_internal_state(const int k) const override;

  ///  set internal state of the material
  void set_internal_state(const int k, const double val) override;

  ///  return number of ionic currents
  int get_number_of_ionic_currents() const override;

  ///  return ionic currents
  double get_ionic_currents(const int k) const override;

  /// time update for this material
  void update(const double phi, const double dt) override;

 private:
  MyocardTools tools_;

  /// perturbation for numerical approximation of the derivative
  double eps_deriv_{};

  /// gating variables Inada
  std::vector<double> s0_;
  std::vector<double> s_;
  std::vector<double> r_;
  std::vector<double> a_;
  std::vector<double> c_;

  double voi_{};  // current time (for debugging with CellML)

};  // Myocard_TenTusscher


FOUR_C_NAMESPACE_CLOSE

#endif
