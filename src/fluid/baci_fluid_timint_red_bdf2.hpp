/*-----------------------------------------------------------*/
/*! \file

\brief BDF-2 time integration for reduced models


\level 2

*/
/*-----------------------------------------------------------*/

#ifndef BACI_FLUID_TIMINT_RED_BDF2_HPP
#define BACI_FLUID_TIMINT_RED_BDF2_HPP


#include "baci_config.hpp"

#include "baci_fluid_timint_bdf2.hpp"
#include "baci_fluid_timint_red.hpp"
#include "baci_linalg_utils_sparse_algebra_math.hpp"

BACI_NAMESPACE_OPEN


namespace FLD
{
  class TimIntRedModelsBDF2 : public TimIntBDF2, public TimIntRedModels
  {
   public:
    /// Standard Constructor
    TimIntRedModelsBDF2(const Teuchos::RCP<DRT::Discretization>& actdis,
        const Teuchos::RCP<CORE::LINALG::Solver>& solver,
        const Teuchos::RCP<Teuchos::ParameterList>& params,
        const Teuchos::RCP<IO::DiscretizationWriter>& output, bool alefluid = false);


    /*!
    \brief initialization

    */
    void Init() override;

    /*!
    \brief read restart data

    */
    void ReadRestart(int step) override;


   protected:
   private:
  };  // class TimIntRedModelsBDF2

}  // namespace FLD


BACI_NAMESPACE_CLOSE

#endif  // FLUID_TIMINT_RED_BDF2_H