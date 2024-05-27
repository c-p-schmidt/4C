/*----------------------------------------------------------------------*/
/*! \file

\brief manages mesh based coupling of fluid and porous media
xfluid class and the cut-library

\level 3

*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_XFEM_COUPLING_FPI_MESH_HPP
#define FOUR_C_XFEM_COUPLING_FPI_MESH_HPP

#include "4C_config.hpp"

#include "4C_discretization_fem_general_utils_boundary_integration.hpp"
#include "4C_discretization_fem_general_utils_gausspoints.hpp"
#include "4C_discretization_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_xfem_coupling_mesh.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CORE::GEO
{
  namespace CUT
  {
    class BoundaryCell;
  }
}  // namespace CORE::GEO

namespace XFEM
{
  /*!
  specialized class for coupling of fluid with an porous media (Darcy Flow)
    \author ager
    \date 06/16
   */
  class MeshCouplingFPI : public MeshVolCoupling
  {
   public:
    enum CoupledField
    {
      ps_ps,
      ps_pf,
      pf_ps,
      pf_pf
    };

    //! constructor
    explicit MeshCouplingFPI(
        Teuchos::RCP<DRT::Discretization>& bg_dis,  ///< background discretization
        const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                       ///< discretization is identified
        Teuchos::RCP<DRT::Discretization>&
            cond_dis,           ///< discretization from which cutter discretization can be derived
        const int coupling_id,  ///< id of composite of coupling conditions
        const double time,      ///< time
        const int step,         ///< time step
        MeshCouplingFPI::CoupledField field  ///< which field is coupled to the fluid
    );

    //! cutter dis should be loaded into the cut?
    bool CutGeometry() override { return (coupled_field_ == MeshCouplingFPI::ps_ps); }

    bool isBJ() const { return full_bj_; }

    void SetFullState(
        Teuchos::RCP<const Epetra_Vector> dispnp, Teuchos::RCP<const Epetra_Vector> pres)
    {
      fulldispnp_ = dispnp;
      fullpres_ = pres;
    }

    void initialize_struc_pres_map(
        Teuchos::RCP<const Epetra_Map> pfmap, Teuchos::RCP<const Epetra_Map> psmap)
    {
      // We need to identify cutter dis dofs and pressure dofs on all processors for the whole
      // cutter_dis, as long as we don't have another ghosting strategy for the cutter_dis ...

      if (pfmap->NumMyElements() != psmap->NumMyElements())
        FOUR_C_THROW(
            "initialize_struc_pres_map: (pfmap->NumGlobalElements() != "
            "psmap->NumGlobalElements())!");

      Teuchos::RCP<Epetra_Map> fullpfmap = CORE::LINALG::AllreduceEMap(*pfmap);
      Teuchos::RCP<Epetra_Map> fullpsmap = CORE::LINALG::AllreduceEMap(*psmap);

      if (fullpfmap->NumMyElements() != fullpsmap->NumMyElements())
        FOUR_C_THROW(
            "initialize_struc_pres_map: (fullpfmap->NumGlobalElements() != "
            "fullpsmap->NumGlobalElements())!");

      for (int lid = 0; lid < fullpfmap->NumMyElements(); ++lid)
        lm_struct_x_lm_pres_[fullpsmap->GID(lid)] =
            fullpfmap->GID(lid) + 1;  // z-component of structure --> pressure
    }

    /*!
     Return prescribed velocities and traction vectors for a GNBC boundary condition.
     Also returns the projection matrix (to the plane of the surface) needed for the GNBC condition.
     */
    template <CORE::FE::CellType DISTYPE, class T1, class M3>
    void evaluate_coupling_conditions(T1& projection_matrix,  ///< Projection matrix
        M3& normal                                            ///< surface normal of cut element
    )
    {
      eval_projection_matrix<DISTYPE>(projection_matrix, normal);
      return;
    };

    // finalize the interface true residual vector
    void complete_state_vectors() override;

    virtual void zero_state_vectors_fpi();

    void GmshOutput(const std::string& filename_base, const int step, const int gmsh_step_diff,
        const bool gmsh_debug_out_screen) override;

    void gmsh_output_discretization(std::ostream& gmshfilecontent) override;

    void LiftDrag(const int step, const double time) const override;

    void read_restart(const int step) override;

    // interface foces
    Teuchos::RCP<Epetra_Vector> ITrueResidual() { return itrueresidual_; }

    // for assembly of fluid interface forces
    Teuchos::RCP<Epetra_Vector> IForcecol() { return iforcecol_; }

    //! Caluculate the Porosity for this FaceElement Gausspoint
    double CalcPorosity(DRT::Element* ele, CORE::LINALG::Matrix<3, 1>& rst_slave, double& J);

    //! get distance when transition between FPSI and PSCI is started
    double get_fpi_pcontact_exchange_dist() { return fpsi_contact_hfraction_ * h_scaling_; }

    //! ration of gap/(POROCONTACTFPSI_HFRACTION*h) when full PSCI is starte
    double get_fpi_pcontact_fullfraction() { return fpsi_contact_fullpcfraction_; }

    /// Assign communicator to contact to mesh coupling object
    void Assign_Contact_Comm(Teuchos::RCP<XFEM::XFluidContactComm> xf_c_comm)
    {
      xf_c_comm_ = xf_c_comm;
    }

    /// Get communicator to contact
    Teuchos::RCP<XFEM::XFluidContactComm> Get_Contact_Comm()
    {
      if (xf_c_comm_ == Teuchos::null)
        FOUR_C_THROW("Get_Contact_Comm: Xfluid_Contact_Communicator not assigned!");
      return xf_c_comm_;
    }

    /// Register this side on this proc
    void RegisterSideProc(int sid);

    /// Reconnect Parent Pointers
    void reconnect_parent_pointers();

    /// Initialize Fluid State
    bool initialize_fluid_state(Teuchos::RCP<CORE::GEO::CutWizard> cutwizard,
        Teuchos::RCP<DRT::Discretization> fluiddis,
        Teuchos::RCP<XFEM::ConditionManager> condition_manager,
        Teuchos::RCP<Teuchos::ParameterList> fluidparams);

   private:
    void Output(const int step, const double time, const bool write_restart_data) override;

   protected:
    //! Initializes configurationmap
    void setup_configuration_map() override;

    //! set the name of the coupling object based on the field coupling
    void SetCouplingName() override;

    //! Updates configurationmap for specific Gausspoint
    void update_configuration_map_gp(double& kappa_m,  //< fluid sided weighting
        double& visc_m,                                //< master sided dynamic viscosity
        double& visc_s,                                //< slave sided dynamic viscosity
        double& density_m,                             //< master sided density
        double& visc_stab_tang,                        //< viscous tangential NIT Penalty scaling
        double& full_stab,                             //< full NIT Penalty scaling
        const CORE::LINALG::Matrix<3, 1>& x,           //< Position x in global coordinates
        const CORE::Conditions::Condition* cond,       //< Condition
        DRT::Element* ele,                             //< Element
        DRT::Element* bele,                            //< Boundary Element
        double* funct,  //< local shape function for Gauss Point (from fluid element)
        double* derxy,  //< local derivatives of shape function for Gauss Point (from fluid element)
        CORE::LINALG::Matrix<3, 1>& rst_slave,  //< local coord of gp on slave boundary element
        CORE::LINALG::Matrix<3, 1>& normal,     //< normal at gp
        CORE::LINALG::Matrix<3, 1>& vel_m,      //< master velocity at gp
        double* fulltraction  //< precomputed fsi traction (sigmaF n + gamma relvel)
        ) override;

    void InitStateVectors() override;

    void do_condition_specific_setup() override;

    bool HasMovingInterface() override { return true; }

    void set_condition_specific_parameters() override;

   private:
    //! Updates configurationmap for specific Gausspoint in the contact case
    virtual void update_configuration_map_gp_contact(double& kappa_m,  //< fluid sided weighting
        double& visc_m,                           //< master sided dynamic viscosity
        double& visc_s,                           //< slave sided dynamic viscosity
        double& density_m,                        //< master sided density
        double& visc_stab_tang,                   //< viscous tangential NIT Penalty scaling
        double& full_stab,                        //< full NIT Penalty scaling
        const CORE::LINALG::Matrix<3, 1>& x,      //< Position x in global coordinates
        const CORE::Conditions::Condition* cond,  //< Condition
        DRT::Element* ele,                        //< Element
        DRT::Element* bele,                       //< Boundary Element
        double* funct,  //< local shape function for Gauss Point (from fluid element)
        double* derxy,  //< local derivatives of shape function for Gauss Point (from fluid element)
        CORE::LINALG::Matrix<3, 1>& rst_slave,  //< local coord of gp on slave boundary element
        CORE::LINALG::Matrix<3, 1>& normal,     //< normal at gp
        CORE::LINALG::Matrix<3, 1>& vel_m,      //< master velocity at gp
        double* fulltraction  //< precomputed fsi traction (sigmaF n + gamma relvel)
    );

    //! Caluculate the Porosity for J,porosity pair on this FaceElement
    double calctr_permeability(DRT::Element* ele, double& porosity, double& J);

    //! Compute Jacobian and extract PoroFluidPressure this FaceElement Gausspoint
    double compute_jacobianand_pressure(
        DRT::Element* ele, CORE::LINALG::Matrix<3, 1>& rst_slave, double& pres);

    //------------------------------- vectors -----------------------------
    //! @name cutter-dis state vectors
    Teuchos::RCP<Epetra_Vector>
        itrueresidual_;  //! interface forces acting on the structural surface (= -iforcenp)
    Teuchos::RCP<Epetra_Vector>
        iforcecol_;  //! interface forces acting on the fluid surface (column vector assembly)
    //@}

    //---------------------------------configuration--------------------------------
    //! @name type of poro field coupled to xfluid handled by this mesh coupling object!
    MeshCouplingFPI::CoupledField coupled_field_;

    //! Full BJ Variant or BJSaffmann?
    bool full_bj_;
    bool sub_tang_;

    Teuchos::RCP<const Epetra_Vector> fulldispnp_;
    Teuchos::RCP<const Epetra_Vector> fullpres_;

    //! map from structural x dof to pres dof of a node!
    std::map<int, int> lm_struct_x_lm_pres_;

    double bj_coeff_;

    //! flag for contact
    bool contact_;

    //! factor of element size, when transition between FPSI and PSCI is started! (if 1,
    //! interpolation within one element)
    double fpsi_contact_hfraction_;
    //! ration of gap/(POROCONTACTFPSI_HFRACTION*h) when full PSCI is started! (if 0, pure contact
    //! starts when gap is zero)
    double fpsi_contact_fullpcfraction_;

    //! Xfluid Contact Communicator
    Teuchos::RCP<XFEM::XFluidContactComm> xf_c_comm_;

    //@}
  };

}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif