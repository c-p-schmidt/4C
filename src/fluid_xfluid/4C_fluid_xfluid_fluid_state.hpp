// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_XFLUID_FLUID_STATE_HPP
#define FOUR_C_FLUID_XFLUID_FLUID_STATE_HPP

#include "4C_config.hpp"

#include "4C_fluid_xfluid_state.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseOperator;
}

namespace XFEM
{
  class ConditionManager;
}

namespace FLD
{
  namespace Utils
  {
    class XFluidFluidMapExtractor;
  }

  /**
   * Container class for the merged state vectors and maps of the intersected background
   * fluid and the embedded (ALE-)fluid.
   */
  class XFluidFluidState : public XFluidState
  {
   public:
    /// ctor
    explicit XFluidFluidState(const Teuchos::RCP<XFEM::ConditionManager>& condition_manager,
        const Teuchos::RCP<Cut::CutWizard>& wizard, const Teuchos::RCP<XFEM::XFEMDofSet>& dofset,
        const Teuchos::RCP<const Epetra_Map>& xfluiddofrowmap,
        const Teuchos::RCP<const Epetra_Map>& xfluiddofcolmap,
        const Teuchos::RCP<const Epetra_Map>& embfluiddofrowmap);

    /// setup map extractors for dirichlet maps & velocity/pressure maps
    void setup_map_extractors(const Teuchos::RCP<Core::FE::Discretization>& xfluiddiscret,
        const Teuchos::RCP<Core::FE::Discretization>& embfluiddiscret, const double& time);

    /// build merged fluid dirichlet map extractor
    void create_merged_dbc_map_extractor(const Core::LinAlg::MapExtractor& embfluiddbcmaps);

    //! @name Accessors
    //@{

    Teuchos::RCP<Core::LinAlg::MapExtractor> dbc_map_extractor() override
    {
      return xffluiddbcmaps_;
    }

    Teuchos::RCP<Core::LinAlg::MapExtractor> vel_pres_splitter() override
    {
      return xffluidvelpressplitter_;
    }

    bool destroy() override;

    Teuchos::RCP<Core::LinAlg::SparseMatrix> system_matrix() override;
    Teuchos::RCP<Core::LinAlg::Vector<double>>& residual() override { return xffluidresidual_; }
    Teuchos::RCP<Core::LinAlg::Vector<double>>& zeros() override { return xffluidzeros_; }
    Teuchos::RCP<Core::LinAlg::Vector<double>>& inc_vel() override { return xffluidincvel_; }
    Teuchos::RCP<Core::LinAlg::Vector<double>>& velnp() override { return xffluidvelnp_; }
    //@}

    void complete_coupling_matrices_and_rhs() override;

    //@name Map of the merged system
    //@{
    /// combined background and embedded fluid dof-map
    Teuchos::RCP<Epetra_Map> xffluiddofrowmap_;
    //@}

    //@name Map extractors of the merged system
    //@{
    /// extractor used for splitting fluid and embedded fluid
    Teuchos::RCP<FLD::Utils::XFluidFluidMapExtractor> xffluidsplitter_;
    /// extractor used for splitting between velocity and pressure dof from the combined background
    /// & embedded fluid dof-map
    Teuchos::RCP<Core::LinAlg::MapExtractor> xffluidvelpressplitter_;
    /// combined background and embedded fluid map extractor for dirichlet-constrained dof
    Teuchos::RCP<Core::LinAlg::MapExtractor> xffluiddbcmaps_;
    //@}

    /// full system matrix for coupled background and embedded fluid
    Teuchos::RCP<Core::LinAlg::SparseOperator> xffluidsysmat_;

    /// a vector of zeros to be used to enforce zero dirichlet boundary conditions
    Teuchos::RCP<Core::LinAlg::Vector<double>> xffluidzeros_;

    /// (standard) residual vector (rhs for the incremental form),
    Teuchos::RCP<Core::LinAlg::Vector<double>> xffluidresidual_;

    //! @name combined background and embedded fluid velocity and pressure at time n+1, n and
    //! increment
    //@{
    /// \f$ \mathbf{u}^{b\cup e,n+1} \f$
    Teuchos::RCP<Core::LinAlg::Vector<double>> xffluidvelnp_;
    /// \f$ \mathbf{u}^{b\cup e,n+1} \f$
    Teuchos::RCP<Core::LinAlg::Vector<double>> xffluidveln_;
    /// \f$ \Delta \mathbf{u}^{b\cup e,n+1}_{i+1} \f$
    Teuchos::RCP<Core::LinAlg::Vector<double>> xffluidincvel_;
    //@}

   private:
    /// initialize all state members based on the merged fluid dof-rowmap
    void init_state_vectors();

    /// initialize the system matrix of the intersected fluid
    void init_system_matrix();

    /// embedded fluid dof-map
    Teuchos::RCP<const Epetra_Map> embfluiddofrowmap_;
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
