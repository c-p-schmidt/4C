/*-----------------------------------------------------------*/
/*! \file



\level 3

*/
/*-----------------------------------------------------------*/

#ifndef FOUR_C_SOLVER_NONLIN_NOX_INNER_STATUSTEST_COMBO_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_INNER_STATUSTEST_COMBO_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_forward_decl.hpp"
#include "4C_solver_nonlin_nox_inner_statustest_generic.hpp"

#include <NOX_StatusTest_Combo.H>
#include <Teuchos_RCP.hpp>

#include <set>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace NLN
  {
    namespace INNER
    {
      namespace StatusTest
      {
        class Combo : public NOX::NLN::INNER::StatusTest::Generic
        {
         public:
          //! Constructor. Optional argument is the error stream for output.
          Combo(::NOX::StatusTest::Combo::ComboType t, const ::NOX::Utils* u = nullptr);

          //! Constructor with a single test.
          Combo(::NOX::StatusTest::Combo::ComboType t, const Teuchos::RCP<Generic>& a,
              const ::NOX::Utils* u = nullptr);

          //! Constructor with two tests.
          Combo(::NOX::StatusTest::Combo::ComboType t, const Teuchos::RCP<Generic>& a,
              const Teuchos::RCP<Generic>& b, const ::NOX::Utils* u = nullptr);

          StatusType CheckStatus(const Interface::Required& interface,
              const ::NOX::Solver::Generic& solver, const ::NOX::Abstract::Group& grp,
              ::NOX::StatusTest::CheckType checkType) override;

          //! Return the result of the most recent inner checkStatus call
          StatusType GetStatus() const override;

          //! Output formatted description of inner stopping test to output stream.
          std::ostream& Print(std::ostream& stream, int indent = 0) const override;

          //! Add another test to this combination.
          /*!
            Calls isSafe() to determine if it is safe to add \c a to the combination.
          */
          Combo& addStatusTest(const Teuchos::RCP<Generic>& a);

          const std::vector<Teuchos::RCP<Generic>>& GetTestVector() const;

         protected:
          //! Check whether or not it is safe to add a to this list of tests.
          bool isSafe(Generic& a);

          /// OR-combination
          void orOp(const Interface::Required& interface, const ::NOX::Solver::Generic& solver,
              const ::NOX::Abstract::Group& grp, ::NOX::StatusTest::CheckType checkType);

          /// AND-combination
          void andOp(const Interface::Required& interface, const ::NOX::Solver::Generic& solver,
              const ::NOX::Abstract::Group& grp, ::NOX::StatusTest::CheckType checkType);

         protected:
          //! Vector of generic status tests
          std::vector<Teuchos::RCP<Generic>> tests_;

          //! Ostream used to print errors
          ::NOX::Utils utils_;

          /// inner test status
          NOX::NLN::INNER::StatusTest::StatusType status_ =
              NOX::NLN::INNER::StatusTest::status_unevaluated;

          /// specified combo type
          ::NOX::StatusTest::Combo::ComboType type_;

          /// set of inner status test results which indicate an unconverged test
          static const std::set<StatusType> unconverged_;

        };  // class Combo
      }     // namespace StatusTest
    }       // namespace INNER
  }         // namespace NLN
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif