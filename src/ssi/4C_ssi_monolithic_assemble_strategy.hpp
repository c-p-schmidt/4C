/*----------------------------------------------------------------------*/
/*! \file
\brief Assemble strategy for monolithic SSI

\level 2

 */
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_SSI_MONOLITHIC_ASSEMBLE_STRATEGY_HPP
#define FOUR_C_SSI_MONOLITHIC_ASSEMBLE_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_ssi_utils.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace CORE::LINALG
{
  class MultiMapExtractor;
  class Solver;
  class SparseMatrix;
}  // namespace CORE::LINALG

namespace SSI
{
  namespace UTILS
  {
    class SSISlaveSideConverter;
  }
  class SsiMono;

  /*!
  We have three options how the global system matrix and the sub matrices are arranged:
  1) System matrix: sparse
    ->Scatra matrix sparse
    ->Structure matrix sparse
  2) System matrix: block
    2a) Scatra matrix block
    ->Structure matrix sparse
    2b) Scatra matrix sparse
    ->Structure matrix sparse

  The inheritance hierarchy is appropriate*/
  class AssembleStrategyBase
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~AssembleStrategyBase() = default;

    //! constructor
    explicit AssembleStrategyBase(
        Teuchos::RCP<const SSI::UTILS::SSIMaps> ssi_maps, const bool is_scatra_manifold);

    //! assemble RHS
    void AssembleRHS(Teuchos::RCP<Epetra_Vector> rhs, Teuchos::RCP<const Epetra_Vector> rhs_scatra,
        Teuchos::RCP<const Epetra_Vector> rhs_structure,
        Teuchos::RCP<const Epetra_Vector> rhs_manifold);

    //! assemble ScaTra-ScaTra-Block into system matrix
    virtual void assemble_scatra_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_scatra_matrix) = 0;

    //! assemble ScaTra-Structure-Block into system matrix
    virtual void assemble_scatra_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_structure_matrix) = 0;

    //! assemble Structure-Structure-Block into system matrix
    virtual void assemble_structure_structure(
        Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseMatrix> structure_structure_matrix) = 0;

    //! assemble Structure-ScaTra-Block into system matrix
    virtual void assemble_structure_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> structure_scatra_matrix) = 0;

    //! assemble ScaTra Manifold-ScaTra Manifold-Block into system matrix
    virtual void assemble_scatramanifold_scatramanifold(
        Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_scatramanifold_matrix) = 0;

    //! assemble ScaTra Manifold-Structure-Block into system matrix
    virtual void assemble_scatramanifold_structure(
        Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_structure_matrix) = 0;

    //! assemble ScaTra Manifold-ScaTra-Block into system matrix
    virtual void assemble_scatramanifold_scatra(
        Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_scatra_matrix) = 0;

    //! assemble ScaTra-ScaTra Manifold-Block into system matrix
    virtual void assemble_scatra_scatramanifold(
        Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_scatramanifold_matrix) = 0;

   protected:
    //! solve additional scatra field on manifolds
    bool IsScaTraManifold() const { return is_scatra_manifold_; }

    //! this object holds all maps relevant to monolithic scalar transport - structure interaction
    Teuchos::RCP<const SSI::UTILS::SSIMaps> SSIMaps() const { return ssi_maps_; }

   private:
    //! solve additional scatra field on manifolds
    const bool is_scatra_manifold_;

    //! this object holds all maps relevant to monolithic scalar transport - structure interaction
    Teuchos::RCP<const SSI::UTILS::SSIMaps> ssi_maps_;
  };

  //======================================================================================================
  // SSI problem is organized in sub matrices
  class AssembleStrategyBlock : public AssembleStrategyBase
  {
   public:
    explicit AssembleStrategyBlock(
        Teuchos::RCP<const SSI::UTILS::SSIMaps> ssi_maps, const bool is_scatra_manifold);

    void assemble_scatra_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_scatra_matrix) override = 0;

    void assemble_scatra_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_structure_matrix) override = 0;

    void assemble_structure_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseMatrix> structure_structure_matrix) override = 0;

    void assemble_structure_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> structure_scatra_matrix) override = 0;

    void assemble_scatramanifold_scatramanifold(
        Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_scatramanifold_matrix)
        override = 0;

    void assemble_scatramanifold_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_structure_matrix) override =
        0;

    void assemble_scatramanifold_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_scatra_matrix) override = 0;

    void assemble_scatra_scatramanifold(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_scatramanifold_matrix) override = 0;

   protected:
    //! position of scatra blocks in system matrix
    const std::vector<int>& BlockPositionScaTra() const { return block_position_scatra_; }

    //! position of scatra manifold blocks in system matrix
    const std::vector<int>& block_position_sca_tra_manifold() const
    {
      return block_position_scatra_manifold_;
    }

    //! position of structure block in system matrix
    int position_structure() const { return position_structure_; };

   private:
    //! position of scatra blocks in system matrix
    const std::vector<int> block_position_scatra_;

    //! position of scatra manifold blocks in system matrix
    std::vector<int> block_position_scatra_manifold_;

    //! position of structure block in system matrix
    const int position_structure_;
  };

  // *********************************************************************************************
  // SSI problem is organized in sparse structure sub matrix and block scatra sub matrix
  class AssembleStrategyBlockBlock : public AssembleStrategyBlock
  {
   public:
    explicit AssembleStrategyBlockBlock(
        Teuchos::RCP<const SSI::UTILS::SSIMaps> ssi_maps, const bool is_scatra_manifold);

    void assemble_scatra_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_scatra_matrix) override;

    void assemble_scatra_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_structure_matrix) override;

    void assemble_structure_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseMatrix> structure_structure_matrix) override;

    void assemble_structure_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> structure_scatra_matrix) override;

    void assemble_scatramanifold_scatramanifold(
        Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_scatramanifold_matrix)
        override;

    void assemble_scatramanifold_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_structure_matrix) override;

    void assemble_scatramanifold_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_scatra_matrix) override;

    void assemble_scatra_scatramanifold(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_scatramanifold_matrix) override;
  };

  // *********************************************************************************************
  // SSI problem is organized in sparse sub matrices
  class AssembleStrategyBlockSparse : public AssembleStrategyBlock
  {
   public:
    explicit AssembleStrategyBlockSparse(
        Teuchos::RCP<const SSI::UTILS::SSIMaps> ssi_maps, const bool is_scatra_manifold);

    void assemble_scatra_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_scatra_matrix) override;

    void assemble_scatra_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_structure_matrix) override;

    void assemble_structure_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseMatrix> structure_structure_matrix) override;

    void assemble_structure_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> structure_scatra_matrix) override;

    void assemble_scatramanifold_scatramanifold(
        Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_scatramanifold_matrix)
        override;

    void assemble_scatramanifold_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_structure_matrix) override;

    void assemble_scatramanifold_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_scatra_matrix) override;

    void assemble_scatra_scatramanifold(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_scatramanifold_matrix) override;
  };

  //======================================================================================================
  // SSI problem is organized in one sparse matrix
  class AssembleStrategySparse : public AssembleStrategyBase
  {
   public:
    explicit AssembleStrategySparse(
        Teuchos::RCP<const SSI::UTILS::SSIMaps> ssi_maps, const bool is_scatra_manifold);

    void assemble_scatra_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_scatra_matrix) override;

    void assemble_scatra_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_structure_matrix) override;

    void assemble_structure_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseMatrix> structure_structure_matrix) override;

    void assemble_structure_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> structure_scatra_matrix) override;

    void assemble_scatramanifold_scatramanifold(
        Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_scatramanifold_matrix)
        override;

    void assemble_scatramanifold_structure(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_structure_matrix) override;

    void assemble_scatramanifold_scatra(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatramanifold_scatra_matrix) override;

    void assemble_scatra_scatramanifold(Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
        Teuchos::RCP<const CORE::LINALG::SparseOperator> scatra_scatramanifold_matrix) override;
  };

  //! build specific assemble strategy
  Teuchos::RCP<SSI::AssembleStrategyBase> BuildAssembleStrategy(
      Teuchos::RCP<const SSI::UTILS::SSIMaps> ssi_maps, const bool is_scatra_manifold,
      CORE::LINALG::MatrixType matrixtype_ssi, CORE::LINALG::MatrixType matrixtype_scatra);

}  // namespace SSI
FOUR_C_NAMESPACE_CLOSE

#endif