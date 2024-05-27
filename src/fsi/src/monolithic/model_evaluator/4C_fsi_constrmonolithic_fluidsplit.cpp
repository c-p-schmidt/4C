/*----------------------------------------------------------------------*/
/*! \file

\brief Solve FSI problem with constraints

\level 2

*/
/*----------------------------------------------------------------------*/
#include "4C_fsi_constrmonolithic_fluidsplit.hpp"

#include "4C_adapter_ale_fsi.hpp"
#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_matrixtransform.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::ConstrMonolithicFluidSplit::ConstrMonolithicFluidSplit(
    const Epetra_Comm& comm, const Teuchos::ParameterList& timeparams)
    : ConstrMonolithic(comm, timeparams)
{
  // ---------------------------------------------------------------------------
  // FSI specific check of Dirichlet boundary conditions
  // ---------------------------------------------------------------------------
  // Create intersection of slave DOFs that hold a Dirichlet boundary condition
  // and are located at the FSI interface
  std::vector<Teuchos::RCP<const Epetra_Map>> intersectionmaps;
  intersectionmaps.push_back(fluid_field()->GetDBCMapExtractor()->CondMap());
  intersectionmaps.push_back(fluid_field()->Interface()->FSICondMap());
  Teuchos::RCP<Epetra_Map> intersectionmap =
      CORE::LINALG::MultiMapExtractor::IntersectMaps(intersectionmaps);

  // Check whether the intersection is empty
  if (intersectionmap->NumGlobalElements() != 0)
  {
    // It is not allowed, that slave DOFs at the interface hold a Dirichlet
    // boundary condition. Thus --> ToDO: Error message

    // We do not have to care whether ALE interface DOFs carry DBCs in the
    // input file since they do not occur in the monolithic system and, hence,
    // do not cause a conflict.

    std::stringstream errormsg;
    errormsg << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl
             << "  |                DIRICHLET BOUNDARY CONDITIONS ON SLAVE SIDE OF FSI INTERFACE   "
                "              |"
             << std::endl
             << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl
             << "  | NOTE: The slave side of the interface is not allowed to carry Dirichlet "
                "boundary conditions.|"
             << std::endl
             << "  |                                                                               "
                "              |"
             << std::endl
             << "  | This is a fluid split scheme. Hence, master and slave field are chosen as "
                "follows:          |"
             << std::endl
             << "  |     MASTER  = STRUCTURE                                                       "
                "              |"
             << std::endl
             << "  |     SLAVE   = FLUID                                                           "
                "              |"
             << std::endl
             << "  |                                                                               "
                "              |"
             << std::endl
             << "  | Dirichlet boundary conditions were detected on slave interface degrees of "
                "freedom. Please   |"
             << std::endl
             << "  | remove Dirichlet boundary conditions from the slave side of the FSI "
                "interface.              |"
             << std::endl
             << "  | Only the master side of the FSI interface is allowed to carry Dirichlet "
                "boundary conditions.|"
             << std::endl
             << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl;

    std::cout << errormsg.str();
  }
  // ---------------------------------------------------------------------------

  scon_t_ =
      Teuchos::rcp(new CORE::LINALG::SparseMatrix(*conman_->GetConstraintMap(), 81, false, true));

  fggtransform_ = Teuchos::rcp(new CORE::LINALG::MatrixRowColTransform);
  fgitransform_ = Teuchos::rcp(new CORE::LINALG::MatrixRowTransform);
  figtransform_ = Teuchos::rcp(new CORE::LINALG::MatrixColTransform);
  fmiitransform_ = Teuchos::rcp(new CORE::LINALG::MatrixColTransform);
  fmgitransform_ = Teuchos::rcp(new CORE::LINALG::MatrixRowColTransform);
  fmigtransform_ = Teuchos::rcp(new CORE::LINALG::MatrixColTransform);
  fmggtransform_ = Teuchos::rcp(new CORE::LINALG::MatrixRowColTransform);
  aigtransform_ = Teuchos::rcp(new CORE::LINALG::MatrixColTransform);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::SetupSystem()
{
  GeneralSetup();

  // create combined map
  create_combined_dof_row_map();

  fluid_field()->UseBlockMatrix(true);

  // build ale system matrix in splitted system
  ale_field()->CreateSystemMatrix(ale_field()->Interface());

  CreateSystemMatrix(false);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::create_combined_dof_row_map()
{
  std::vector<Teuchos::RCP<const Epetra_Map>> vecSpaces;
  vecSpaces.push_back(StructureField()->dof_row_map());
  vecSpaces.push_back(fluid_field()->dof_row_map());
  vecSpaces.push_back(ale_field()->Interface()->OtherMap());
  vecSpaces.push_back(conman_->GetConstraintMap());

  if (vecSpaces[0]->NumGlobalElements() == 0)
    FOUR_C_THROW("No inner structural equations. Splitting not possible. Panic.");

  set_dof_row_maps(vecSpaces);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::setup_rhs_residual(Epetra_Vector& f)
{
  const double scale = fluid_field()->ResidualScaling();

  setup_vector(f, StructureField()->RHS(), fluid_field()->RHS(), ale_field()->RHS(),
      conman_->GetError(), scale);

  // add additional ale residual
  Extractor().AddVector(*aleresidual_, 2, f);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::setup_rhs_lambda(Epetra_Vector& f)
{
  // ToDo: We still need to implement this.

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::setup_rhs_firstiter(Epetra_Vector& f)
{
  Teuchos::RCP<const CORE::LINALG::BlockSparseMatrixBase> blockf =
      fluid_field()->BlockSystemMatrix();

  const CORE::LINALG::SparseMatrix& fig = blockf->Matrix(0, 1);
  const CORE::LINALG::SparseMatrix& fgg = blockf->Matrix(1, 1);

  Teuchos::RCP<const Epetra_Vector> fveln = fluid_field()->extract_interface_veln();
  const double timescale = fluid_field()->TimeScaling();
  const double scale = fluid_field()->ResidualScaling();

  Teuchos::RCP<Epetra_Vector> rhs = Teuchos::rcp(new Epetra_Vector(fig.RowMap()));

  fig.Apply(*fveln, *rhs);
  rhs->Scale(timescale * Dt());

  rhs = fluid_field()->Interface()->InsertOtherVector(rhs);
  Extractor().AddVector(*rhs, 1, f);

  rhs = Teuchos::rcp(new Epetra_Vector(fgg.RowMap()));

  fgg.Apply(*fveln, *rhs);
  rhs->Scale(scale * timescale * Dt());

  rhs = FluidToStruct(rhs);
  rhs = StructureField()->Interface()->InsertFSICondVector(rhs);
  Extractor().AddVector(*rhs, 0, f);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::setup_system_matrix(CORE::LINALG::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicFluidSplit::setup_system_matrix");

  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here. Extract Jacobian matrices and put them into composite system
  // matrix W

  const CORE::ADAPTER::Coupling& coupsf = structure_fluid_coupling();
  const CORE::ADAPTER::Coupling& coupsa = structure_ale_coupling();
  const CORE::ADAPTER::Coupling& coupfa = FluidAleCoupling();

  /*----------------------------------------------------------------------*/

  double scale = fluid_field()->ResidualScaling();
  double timescale = fluid_field()->TimeScaling();

  Teuchos::RCP<CORE::LINALG::BlockSparseMatrixBase> blockf = fluid_field()->BlockSystemMatrix();


  CORE::LINALG::SparseMatrix& fii = blockf->Matrix(0, 0);
  CORE::LINALG::SparseMatrix& fig = blockf->Matrix(0, 1);
  CORE::LINALG::SparseMatrix& fgi = blockf->Matrix(1, 0);
  CORE::LINALG::SparseMatrix& fgg = blockf->Matrix(1, 1);
  /*----------------------------------------------------------------------*/

  Teuchos::RCP<CORE::LINALG::BlockSparseMatrixBase> a = ale_field()->BlockSystemMatrix();

  if (a == Teuchos::null) FOUR_C_THROW("expect ale block matrix");

  CORE::LINALG::SparseMatrix& aii = a->Matrix(0, 0);
  CORE::LINALG::SparseMatrix& aig = a->Matrix(0, 1);
  /*----------------------------------------------------------------------*/
  // structure part

  Teuchos::RCP<CORE::LINALG::SparseMatrix> s = StructureField()->SystemMatrix();
  // const std::string fname = "cfsstructmatrix.mtl";
  // CORE::LINALG::PrintMatrixInMatlabFormat(fname,*(s->EpetraMatrix()));


  /*----------------------------------------------------------------------*/
  // structure constraint part
  Teuchos::RCP<CORE::LINALG::SparseOperator> tmp = conman_->GetConstrMatrix();
  CORE::LINALG::SparseMatrix scon = *(Teuchos::rcp_dynamic_cast<CORE::LINALG::SparseMatrix>(tmp));

  scon.Complete(*conman_->GetConstraintMap(), s->RangeMap());

  mat.Assign(0, 3, CORE::LINALG::View, scon);

  /*----------------------------------------------------------------------*/
  // fluid part

  //    // uncomplete because the fluid interface can have more connections than the
  // structural one. (Tet elements in fluid can cause this.) We should do
  //   this just once...
  s->UnComplete();


  (*fggtransform_)(fgg, scale * timescale, CORE::ADAPTER::CouplingSlaveConverter(coupsf),
      CORE::ADAPTER::CouplingSlaveConverter(coupsf), *s, true, true);

  mat.Assign(0, 0, CORE::LINALG::View, *s);

  (*fgitransform_)(fgi, scale, CORE::ADAPTER::CouplingSlaveConverter(coupsf), mat.Matrix(0, 1));

  (*figtransform_)(blockf->FullRowMap(), blockf->FullColMap(), fig, timescale,
      CORE::ADAPTER::CouplingSlaveConverter(coupsf), mat.Matrix(1, 0));

  mat.Matrix(1, 1).Add(fii, false, 1., 0.0);
  Teuchos::RCP<CORE::LINALG::SparseMatrix> eye =
      CORE::LINALG::Eye(*fluid_field()->Interface()->FSICondMap());
  mat.Matrix(1, 1).Add(*eye, false, 1., 1.0);

  (*aigtransform_)(a->FullRowMap(), a->FullColMap(), aig, 1.,
      CORE::ADAPTER::CouplingSlaveConverter(coupsa), mat.Matrix(2, 0));
  mat.Assign(2, 2, CORE::LINALG::View, aii);

  /*----------------------------------------------------------------------*/
  // add optional fluid linearization with respect to mesh motion block

  Teuchos::RCP<CORE::LINALG::BlockSparseMatrixBase> mmm = fluid_field()->ShapeDerivatives();
  if (mmm != Teuchos::null)
  {
    CORE::LINALG::SparseMatrix& fmii = mmm->Matrix(0, 0);
    CORE::LINALG::SparseMatrix& fmgi = mmm->Matrix(1, 0);

    CORE::LINALG::SparseMatrix& fmig = mmm->Matrix(0, 1);
    CORE::LINALG::SparseMatrix& fmgg = mmm->Matrix(1, 1);

    // reuse transform objects to add shape derivative matrices to structural blocks

    (*figtransform_)(blockf->FullRowMap(), blockf->FullColMap(), fmig, 1.,
        CORE::ADAPTER::CouplingSlaveConverter(coupsf), mat.Matrix(1, 0), false, true);

    (*fggtransform_)(fmgg, scale, CORE::ADAPTER::CouplingSlaveConverter(coupsf),
        CORE::ADAPTER::CouplingSlaveConverter(coupsf), mat.Matrix(0, 0), false, true);

    // We cannot copy the pressure value. It is not used anyway. So no exact
    // match here.
    (*fmiitransform_)(mmm->FullRowMap(), mmm->FullColMap(), fmii, 1.,
        CORE::ADAPTER::CouplingMasterConverter(coupfa), mat.Matrix(1, 2), false);

    (*fmgitransform_)(fmgi, scale, CORE::ADAPTER::CouplingSlaveConverter(coupsf),
        CORE::ADAPTER::CouplingMasterConverter(coupfa), mat.Matrix(0, 2), false, false);
  }
  /*----------------------------------------------------------------------*/
  // constraint part -> structure

  scon = *(Teuchos::rcp_dynamic_cast<CORE::LINALG::SparseMatrix>(tmp));

  scon_t_->Add(scon, true, 1.0, 0.0);
  scon_t_->Complete(scon.RangeMap(), scon.DomainMap());
  mat.Assign(3, 0, CORE::LINALG::View, *scon_t_);

  /*----------------------------------------------------------------------*/
  // done. make sure all blocks are filled.
  mat.Complete();

  // Finally, take care of Dirichlet boundary conditions
  mat.ApplyDirichlet(*(dbcmaps_->CondMap()), true);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::initial_guess(Teuchos::RCP<Epetra_Vector> ig)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicFluidSplit::initial_guess");

  Teuchos::RCP<Epetra_Vector> ConstraintInitialGuess =
      Teuchos::rcp(new Epetra_Vector(*(conman_->GetConstraintMap()), true));

  setup_vector(*ig, StructureField()->initial_guess(), fluid_field()->initial_guess(),
      ale_field()->initial_guess(), ConstraintInitialGuess, 0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::setup_vector(Epetra_Vector& f,
    Teuchos::RCP<const Epetra_Vector> sv, Teuchos::RCP<const Epetra_Vector> fv,
    Teuchos::RCP<const Epetra_Vector> av, Teuchos::RCP<const Epetra_Vector> cv, double fluidscale)
{
  // extract the inner and boundary dofs of all three fields

  Teuchos::RCP<Epetra_Vector> fov = fluid_field()->Interface()->ExtractOtherVector(fv);
  fov = fluid_field()->Interface()->InsertOtherVector(fov);
  Teuchos::RCP<Epetra_Vector> aov = ale_field()->Interface()->ExtractOtherVector(av);

  if (fabs(fluidscale) >= 1.0E-10)
  {
    // add fluid interface values to structure vector
    Teuchos::RCP<Epetra_Vector> fcv = fluid_field()->Interface()->ExtractFSICondVector(fv);
    Teuchos::RCP<Epetra_Vector> modsv =
        StructureField()->Interface()->InsertFSICondVector(FluidToStruct(fcv));
    modsv->Update(1.0, *sv, fluidscale);

    Extractor().InsertVector(*modsv, 0, f);
  }
  else
  {
    Extractor().InsertVector(*sv, 0, f);
  }

  Extractor().InsertVector(*fov, 1, f);
  Extractor().InsertVector(*aov, 2, f);
  Epetra_Vector modcv = *cv;
  modcv.Scale(-1.0);
  Extractor().InsertVector(modcv, 3, f);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::extract_field_vectors(Teuchos::RCP<const Epetra_Vector> x,
    Teuchos::RCP<const Epetra_Vector>& sx, Teuchos::RCP<const Epetra_Vector>& fx,
    Teuchos::RCP<const Epetra_Vector>& ax)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::ConstrMonolithicFluidSplit::extract_field_vectors");

  // We have overlap at the interface. Thus we need the interface part of the
  // structure vector and append it to the fluid and ale vector. (With the
  // right translation.)

  sx = Extractor().ExtractVector(x, 0);
  Teuchos::RCP<const Epetra_Vector> scx = StructureField()->Interface()->ExtractFSICondVector(sx);

  // process fluid unknowns

  Teuchos::RCP<const Epetra_Vector> fox = Extractor().ExtractVector(x, 1);
  fox = fluid_field()->Interface()->ExtractOtherVector(fox);
  Teuchos::RCP<Epetra_Vector> fcx = StructToFluid(scx);

  fluid_field()->displacement_to_velocity(fcx);

  Teuchos::RCP<Epetra_Vector> f = fluid_field()->Interface()->InsertOtherVector(fox);
  fluid_field()->Interface()->InsertFSICondVector(fcx, f);
  fx = f;

  // process ale unknowns

  Teuchos::RCP<const Epetra_Vector> aox = Extractor().ExtractVector(x, 2);
  Teuchos::RCP<Epetra_Vector> acx = StructToAle(scx);
  Teuchos::RCP<Epetra_Vector> a = ale_field()->Interface()->InsertOtherVector(aox);
  ale_field()->Interface()->InsertVector(acx, 1, a);

  ax = a;
}

FOUR_C_NAMESPACE_CLOSE