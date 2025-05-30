// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_LINEBASED_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_LINEBASED_HPP

#include "4C_config.hpp"

#include "4C_inpar_bio.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_nonconforming.hpp"


FOUR_C_NAMESPACE_OPEN

namespace PoroPressureBased
{
  class PoroMultiPhaseScatraArteryCouplingPairBase;

  //! Line-based coupling between artery network and porofluid-elasticity-scatra algorithm
  class PorofluidElastScatraArteryCouplingLineBasedAlgorithm
      : public PorofluidElastScatraArteryCouplingNonConformingAlgorithm
  {
   public:
    PorofluidElastScatraArteryCouplingLineBasedAlgorithm(
        std::shared_ptr<Core::FE::Discretization> arterydis,
        std::shared_ptr<Core::FE::Discretization> contdis,
        const Teuchos::ParameterList& couplingparams, const std::string& condname,
        const std::string& artcoupleddofname, const std::string& contcoupleddofname);

    //! set-up linear system of equations of coupled problem
    void setup_system(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_cont,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_art,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_cont,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_art,
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_cont,
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_art) override;

    //! setup the strategy
    void setup() override;

    //! apply mesh movement (on artery elements)
    void apply_mesh_movement() override;

    //! access to blood vessel volume fraction
    std::shared_ptr<const Core::LinAlg::Vector<double>> blood_vessel_volume_fraction() override;

   private:
    //! pre-evaluate the pairs and sort out duplicates
    void pre_evaluate_coupling_pairs();

    //! fill the length not changed through deformation and initialize curr length
    void fill_unaffected_artery_length();

    //! fill the integrated diameter not changed through varying blood vessel diameter
    void fill_unaffected_integrated_diam();

    //! calculate the volume fraction occupied by blood vessels
    void calculate_blood_vessel_volume_fraction();

    //! create the GID to segment vector
    void create_gid_to_segment_vector();

    //! fill the GID to segment vector
    void fill_gid_to_segment_vector(
        const std::vector<std::shared_ptr<
            PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase>>& coupled_ele_pairs,
        std::map<int, std::vector<double>>& gid_to_seglength);

    //! set the artery diameter in column based vector
    void fill_artery_ele_diam_col();

    //! (re-)set the artery diameter in material to be able to use it on 1D discretization
    void set_artery_diameter_in_material() override;

    //! reset the integrated diameter vector to zero
    void reset_integrated_diameter_to_zero() override;

    /*!
     * @brief Utility function for depth-first search for the connected components of the 1D artery
     * discretization
     *
     * @param actnode : currently checked node
     * @param visited : vector that marks visited nodes
     * @param artsearchdis : artery-discretization in fully-overlapping format
     * @param ele_diams_artery_full_overlap : vector of element diameters in fully-overlapping
     * format
     * @param this_connected_comp : current connected component
     */
    void depth_first_search_util(Core::Nodes::Node* actnode,
        std::shared_ptr<Core::LinAlg::Vector<int>> visited,
        std::shared_ptr<Core::FE::Discretization> artconncompdis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> ele_diams_artery_full_overlap,
        std::vector<int>& this_connected_comp);

    /*!
     * @brief find free-hanging 1D elements
     *
     * Find the free hanging 1D elements which have to be taken out of simulation during blood
     * vessel collapse. This is realized by getting the connected components of the 1D graph. If no
     * node of a connected component has a DBC or if it is smaller than a user-specified
     * threshold, all its elements are taken out
     * @param : eles_to_be_deleted vector of free-hanging elements
     */
    void find_free_hanging_1d_elements(std::vector<int>& eles_to_be_deleted);

    //! evaluate additional linearization of (integrated) element diameter dependent terms
    //! (Hagen-Poiseuille)
    void evaluate_additional_linearizationof_integrated_diam() override;

    /*!
     * @brief apply additional Dirichlet boundary condition for collapsed 1D elements to avoid
     * singular stiffness matrix
     *
     * apply additional dirichlet boundary conditions of zero pressure or mass fraction on nodes
     * which only border collapsed 1D elements, i.e., free-hanging nodes with zero row in global
     * stiffness matrix to avoid singularity of this matrix
     * \note this procedure is equivalent to taking collapsed elements out of the simulation
     * entirely
     *
     * @param[in]        dbcmap_art map of nodes with DBC of 1D discretization
     * @param[in,out]   rhs_art_with_collapsed right hand side of artery subpart
     * @returns dbcmap, also containing additional boundary condition for collapsed eles
     */
    std::shared_ptr<Core::LinAlg::Map> get_additional_dbc_for_collapsed_eles(
        const Core::LinAlg::MapExtractor& dbcmap_art,
        Core::LinAlg::Vector<double>& rhs_art_with_collapsed);

    //! FE-assemble into global force and stiffness
    void assemble(const int& ele1_gid, const int& ele2_gid, const double& integrated_diameter,
        std::vector<Core::LinAlg::SerialDenseVector> const& ele_rhs,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& ele_matrix,
        std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs) override;

    //! get the segment lengths of element 'artelegid'
    std::vector<double> get_ele_segment_lengths(const int artelegid) override;

    //! check for duplicate segment
    bool is_duplicate_segment(
        const std::vector<std::shared_ptr<
            PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase>>& coupled_ele_pairs,
        PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase& possible_duplicate);

    //! check for identical segment
    bool is_identical_segment(
        const std::vector<std::shared_ptr<
            PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase>>& coupled_ele_pairs,
        const int& ele1gid, const double& etaA, const double& etaB, int& ele_pair_id);

    //! set flag if variable diameter has to be calculated
    void set_flag_variable_diameter() override;

    //! print output of mesh tying pairs
    void output_summary() const;

    //! print out the coupling method
    void print_coupling_method() const override;

    //! maximum number of segments per artery element
    int maxnumsegperartele_;

    //! length of artery elements unaffected by deformation
    std::shared_ptr<Epetra_FEVector> unaffected_seg_lengths_artery_;

    //! length of artery elements in current configuration
    std::shared_ptr<Epetra_FEVector> current_seg_lengths_artery_;

    //! diameter of the artery element integrated over the length of the artery element (row format
    //! and FE vector due to non-local assembly)
    std::shared_ptr<Epetra_FEVector> integrated_diams_artery_row_;

    //! diameter of artery element integrated over the length of the artery element (col format)
    std::shared_ptr<Core::LinAlg::Vector<double>> integrated_diams_artery_col_;

    //! diameter of artery element (col format)
    std::shared_ptr<Core::LinAlg::Vector<double>> ele_diams_artery_col_;

    //! unaffected diameter integrated over the length of the artery element
    //! (protruding elements for which diameter does not change)
    std::shared_ptr<Core::LinAlg::Vector<double>> unaffected_integrated_diams_artery_col_;

    //! volume fraction of blood vessels (for output)
    std::shared_ptr<Core::LinAlg::Vector<double>> bloodvesselvolfrac_;

    //! gid to segment: stores [GID; [eta_a eta_b]_1, [eta_a eta_b]_2, ..., [eta_a eta_b]_n]
    //!  of artery elements in column format, i.e. fully overlapping
    std::map<int, std::vector<double>> gid_to_segment_;

    //! gid to segment length: stores [GID; seglength_1, seglength_2, ..., seglength_n]
    //!  of artery elements in column format, i.e. fully overlapping (only used for
    //!  porofluid-problems)
    std::map<int, std::vector<double>> gid_to_seglength_;
  };
}  // namespace PoroPressureBased

FOUR_C_NAMESPACE_CLOSE

#endif
