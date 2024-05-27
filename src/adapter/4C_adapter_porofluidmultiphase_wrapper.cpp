/*----------------------------------------------------------------------*/
/*! \file
 \brief a wrapper for porous multiphase flow algorithms

   \level 3

 *----------------------------------------------------------------------*/

#include "4C_adapter_porofluidmultiphase_wrapper.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_validparameters.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_porofluidmultiphase_timint_implicit.hpp"
#include "4C_porofluidmultiphase_timint_ost.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ADAPTER::PoroFluidMultiphaseWrapper::PoroFluidMultiphaseWrapper(
    Teuchos::RCP<PoroFluidMultiphase> porofluid)
    : porofluid_(porofluid)
{
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::Init(const bool isale,  ///< ALE flag
    const int nds_disp,           ///< number of dofset associated with displacements
    const int nds_vel,            ///< number of dofset associated with fluid velocities
    const int nds_solidpressure,  ///< number of dofset associated with solid pressure
    const int
        ndsporofluid_scatra,  ///< number of dofset associated with scalar on fluid discretization
    const std::map<int, std::set<int>>* nearbyelepairs  ///< possible interaction partners between
                                                        ///< porofluid and artery discretization
)
{
  // initialize algorithm for specific time-integration scheme
  porofluid_->Init(
      isale, nds_disp, nds_vel, nds_solidpressure, ndsporofluid_scatra, nearbyelepairs);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<CORE::UTILS::ResultTest> ADAPTER::PoroFluidMultiphaseWrapper::CreateFieldTest()
{
  return porofluid_->CreateFieldTest();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> ADAPTER::PoroFluidMultiphaseWrapper::dof_row_map(unsigned nds) const
{
  return porofluid_->dof_row_map(nds);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> ADAPTER::PoroFluidMultiphaseWrapper::ArteryDofRowMap() const
{
  return porofluid_->ArteryDofRowMap();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::BlockSparseMatrixBase>
ADAPTER::PoroFluidMultiphaseWrapper::artery_porofluid_sysmat() const
{
  return porofluid_->artery_porofluid_sysmat();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<DRT::Discretization> ADAPTER::PoroFluidMultiphaseWrapper::Discretization() const
{
  return porofluid_->Discretization();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::read_restart(int restart)
{
  return porofluid_->read_restart(restart);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::ApplyMeshMovement(
    Teuchos::RCP<const Epetra_Vector> dispnp  //!< displacement vector
)
{
  porofluid_->ApplyMeshMovement(dispnp);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::SetVelocityField(Teuchos::RCP<const Epetra_Vector> vel)
{
  porofluid_->SetVelocityField(vel);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::set_state(
    unsigned nds, const std::string& name, Teuchos::RCP<const Epetra_Vector> state)
{
  porofluid_->set_state(nds, name, state);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::SetScatraSolution(
    unsigned nds, Teuchos::RCP<const Epetra_Vector> scalars)
{
  set_state(nds, "scalars", scalars);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::Phinp() const
{
  return porofluid_->Phinp();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::Phin() const
{
  return porofluid_->Phin();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::SolidPressure() const
{
  return porofluid_->SolidPressure();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::Pressure() const
{
  return porofluid_->Pressure();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::Saturation() const
{
  return porofluid_->Saturation();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::valid_vol_frac_spec_dofs()
    const
{
  return porofluid_->valid_vol_frac_spec_dofs();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_MultiVector> ADAPTER::PoroFluidMultiphaseWrapper::Flux() const
{
  return porofluid_->Flux();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int ADAPTER::PoroFluidMultiphaseWrapper::get_dof_set_number_of_solid_pressure() const
{
  return porofluid_->get_dof_set_number_of_solid_pressure();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::TimeLoop() { porofluid_->TimeLoop(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::prepare_time_step() { porofluid_->prepare_time_step(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::Output() { porofluid_->Output(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::Update() { porofluid_->Update(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::evaluate_error_compared_to_analytical_sol()
{
  porofluid_->evaluate_error_compared_to_analytical_sol();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::Solve() { porofluid_->Solve(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::prepare_time_loop() { porofluid_->prepare_time_loop(); }
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const CORE::LINALG::MapExtractor>
ADAPTER::PoroFluidMultiphaseWrapper::GetDBCMapExtractor() const
{
  return porofluid_->GetDBCMapExtractor();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::RHS() const
{
  return porofluid_->RHS();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::ArteryPorofluidRHS() const
{
  return porofluid_->ArteryPorofluidRHS();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::UpdateIter(const Teuchos::RCP<const Epetra_Vector> inc)
{
  porofluid_->UpdateIter(inc);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::reconstruct_pressures_and_saturations()
{
  porofluid_->reconstruct_pressures_and_saturations();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::ReconstructFlux() { porofluid_->ReconstructFlux(); }
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::calculate_phase_velocities()
{
  porofluid_->calculate_phase_velocities();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::Evaluate() { porofluid_->Evaluate(); }
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::SparseMatrix> ADAPTER::PoroFluidMultiphaseWrapper::SystemMatrix()
{
  return porofluid_->SystemMatrix();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::assemble_fluid_struct_coupling_mat(
    Teuchos::RCP<CORE::LINALG::SparseOperator> k_fs)
{
  return porofluid_->assemble_fluid_struct_coupling_mat(k_fs);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::assemble_fluid_scatra_coupling_mat(
    Teuchos::RCP<CORE::LINALG::SparseOperator> k_pfs)
{
  return porofluid_->assemble_fluid_scatra_coupling_mat(k_pfs);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<ADAPTER::ArtNet> ADAPTER::PoroFluidMultiphaseWrapper::ArtNetTimInt()
{
  return porofluid_->ArtNetTimInt();
}

FOUR_C_NAMESPACE_CLOSE