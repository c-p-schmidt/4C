// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_RESULTTEST_HPP
#define FOUR_C_FSI_RESULTTEST_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_utils_result_test.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace FSI
{
  // forward declarations
  class Monolithic;
  class MonolithicNoNOX;

  /*!
    \brief FSI specific result test class

    Here, additional result tests of quantities that do not belong to a single
    field are tested. Basically, this should work for monolithic and partitioned
    schemes as well.

    Feel free to add further testing functionalities!

    \sa ResultTest

  */
  class FSIResultTest : public Core::Utils::ResultTest
  {
   public:
    //! constructor for standard FSI
    FSIResultTest(std::shared_ptr<FSI::Monolithic>&
                      fsi,  ///< monolithic solver object that was used for the simulation
        const Teuchos::ParameterList& fsidyn  ///< FSI parameter list from input file
    );

    //! constructor for FSI implementation without NOX
    FSIResultTest(
        std::shared_ptr<FSI::MonolithicNoNOX>
            fsi,  ///< monolithic solver object without NOX that was used for the simulation
        const Teuchos::ParameterList& fsidyn  ///< FSI parameter list from input file
    );

    //! \brief fsi version of nodal value tests
    //!
    //! Possible position flags are "lambdax", "lambday", "lambdaz"
    void test_node(
        const Core::IO::InputParameterContainer&
            container,   ///< container with expected results as specified in the input file
        int& nerr,       ///< number of tests with errors
        int& test_count  ///< number of tests performed
        ) override;

    //! \brief fsi version of element value tests
    void test_element(
        const Core::IO::InputParameterContainer&
            container,   ///< container with expected results as specified in the input file
        int& nerr,       ///< number of tests with errors
        int& test_count  ///< number of tests performed
        ) override;


    //! \brief fsi version of special tests
    void test_special(
        const Core::IO::InputParameterContainer&
            container,   ///< container with expected results as specified in the input file
        int& nerr,       ///< number of tests with errors
        int& test_count  ///< number of tests performed
        ) override;

   private:
    //! slave discretisation
    std::shared_ptr<Core::FE::Discretization> slavedisc_;

    //! Lagrange multiplier living on the slave discretization
    std::shared_ptr<Core::LinAlg::Vector<double>> fsilambda_;

    //! the monolithic solver object itself
    std::shared_ptr<FSI::Monolithic> fsi_;
  };
}  // namespace FSI
FOUR_C_NAMESPACE_CLOSE

#endif
