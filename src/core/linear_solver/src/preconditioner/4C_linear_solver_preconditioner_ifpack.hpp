/*----------------------------------------------------------------------------*/
/*! \file

\brief CORE::LINALG::SOLVER wrapper around Trilinos' IFPACK preconditioner

\level 0

*/
/*----------------------------------------------------------------------------*/
#ifndef FOUR_C_LINEAR_SOLVER_PRECONDITIONER_IFPACK_HPP
#define FOUR_C_LINEAR_SOLVER_PRECONDITIONER_IFPACK_HPP

#include "4C_config.hpp"

#include "4C_linear_solver_preconditioner_type.hpp"

#include <Ifpack.h>

FOUR_C_NAMESPACE_OPEN

namespace CORE::LINEAR_SOLVER
{
  /*! \brief  IFPACK preconditioners
   *
   *  Set of standard single-matrix preconditioners.
   */
  class IFPACKPreconditioner : public LINEAR_SOLVER::PreconditionerTypeBase
  {
   public:
    //! Constructor (empty)
    IFPACKPreconditioner(Teuchos::ParameterList& ifpacklist, Teuchos::ParameterList& solverlist);

    //! Setup
    void Setup(bool create, Epetra_Operator* matrix, Epetra_MultiVector* x,
        Epetra_MultiVector* b) override;

   private:
    //! IFPACK parameter list
    Teuchos::ParameterList& ifpacklist_;

    //! solver parameter list
    Teuchos::ParameterList& solverlist_;

    //! system of equations used for preconditioning used by P_ only
    Teuchos::RCP<Epetra_RowMatrix> pmatrix_;

  };  // class IFPACKPreconditioner
}  // namespace CORE::LINEAR_SOLVER

FOUR_C_NAMESPACE_CLOSE

#endif