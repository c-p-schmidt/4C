/*----------------------------------------------------------------------*/
/*! \file

\brief singleton class holding all static time integration parameters required for scalar transport
element evaluation

This singleton class holds all static time integration parameters required for scalar transport
element evaluation. All parameters are usually set only once at the beginning of a simulation,
namely during initialization of the global time integrator, and then never touched again throughout
the simulation. This parameter class needs to coexist with the general parameter class holding all
general static parameters required for scalar transport element evaluation.


\level 1
*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_SCATRA_ELE_PARAMETER_TIMINT_HPP
#define FOUR_C_SCATRA_ELE_PARAMETER_TIMINT_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_parameter_base.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace DRT
{
  namespace ELEMENTS
  {
    /// Evaluation of general parameters (constant over time)
    class ScaTraEleParameterTimInt : public ScaTraEleParameterBase
    {
     public:
      //! singleton access method
      static ScaTraEleParameterTimInt* Instance(const std::string& disname);

      //! set parameters
      void SetParameters(Teuchos::ParameterList& parameters) override;

      bool IsGenAlpha() const { return is_genalpha_; };
      bool IsStationary() const { return is_stationary_; };
      bool IsIncremental() const { return is_incremental_; };
      double Time() const { return time_; };
      double TimeDerivativeFac() const { return timederivativefac_; }
      double Dt() const { return dt_; };
      double TimeFac() const { return timefac_; };
      double TimeFacRhs() const { return timefacrhs_; };
      double TimeFacRhsTau() const { return timefacrhstau_; };
      double AlphaF() const { return alpha_f_; };

     private:
      //! private constructor for singletons
      ScaTraEleParameterTimInt();

      bool is_genalpha_;
      bool is_stationary_;
      bool is_incremental_;
      double time_;
      double timederivativefac_;
      double dt_;
      double timefac_;
      double timefacrhs_;
      double timefacrhstau_;
      double alpha_f_;
    };  // class ScaTraEleParameterTimInt
  }     // namespace ELEMENTS
}  // namespace DRT
FOUR_C_NAMESPACE_CLOSE

#endif