/*----------------------------------------------------------------------*/
/*! \file

\brief manages the different types of level-set based coupling conditions and thereby builds the
bridge between the xfluid class and the cut-library

\level 2

*/
/*----------------------------------------------------------------------*/

#include "4C_xfem_coupling_levelset.hpp"

#include "4C_cut_cutwizard.hpp"
#include "4C_discretization_condition_utils.hpp"
#include "4C_discretization_fem_general_extract_values.hpp"
#include "4C_discretization_fem_general_l2_projection.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_gmsh.hpp"
#include "4C_io_pstream.hpp"
#include "4C_lib_discret_xfem.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_utils_function.hpp"
#include "4C_xfem_interface_utils.hpp"

FOUR_C_NAMESPACE_OPEN

// TODO: CouplingBase should become abstract class


XFEM::LevelSetCoupling::LevelSetCoupling(
    Teuchos::RCP<DRT::Discretization>& bg_dis,  ///< background discretization
    const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                   ///< discretization is identified
    Teuchos::RCP<DRT::Discretization>&
        cond_dis,           ///< full discretization from which the cutter discretization is derived
    const int coupling_id,  ///< id of composite of coupling conditions
    const double time,      ///< time
    const int step          ///< time step
    )
    : CouplingBase(bg_dis, cond_name, cond_dis, coupling_id, time, step),
      bg_nds_phi_(-1),
      cutter_nds_phi_(-1),
      normal_orientation_(-1.0),  // level set gradient is directed from inside to outside -- normal
                                  // in xfluid points from outside to inside
      have_nodematching_dis_(false)
{
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::SetCouplingDofsets()
{
  bg_nds_phi_ = get_coupling_dofset_nds("phi_scatra_proxy_in_fluid");
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::SetConditionsToCopy()
{
  // set only the unique given condition name
  conditions_to_copy_.push_back(cond_name_);

  // additional conditions required for the levelset field based on the cutter (background) mesh
  conditions_to_copy_.push_back("XFEMSurfDisplacement");
}


// TODO: needs to be generalized!
/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::set_cutter_discretization()
{
  if (&*cond_dis_ != &*bg_dis_)  // check not required for two-phase anymore
    FOUR_C_THROW(
        "for non-twophase couplings, we have not checked the functionality of using an extra "
        "scatra this -- lets try!");

  /// level-set field is given w.r.t background mesh
  /// NOTE: more generally it would be possible cutterdis != bg_dis for the single LevelSetCoupling,
  /// however, the unique bg_phinp vector stored in the ConditionManager has to be given w.r.t bgdis

  // TODO: shall we allow to use subsets of the cond_dis_ as a cutterdis?
  cutter_dis_ = cond_dis_;

  // do we have node-matching disretizations? otherwise we need to somehow project quantities
  // between the meshes
  have_nodematching_dis_ = HaveMatchingNodes(cutter_dis_, bg_dis_);


  std::string dofset_name = "";

  if (cutter_dis_->Name() == "scatra")
    dofset_name = "phi_in_scatra";
  else if (cutter_dis_->Name() == "fluid")
    dofset_name = "phi_scatra_proxy_in_fluid";
  else
    FOUR_C_THROW("unsupported cutter dis!");

  if (not(dofset_coupling_map_.count(dofset_name) > 0))
    FOUR_C_THROW("dofset not set in dofset_coupling_map for cutter dis!");

  cutter_nds_phi_ = dofset_coupling_map_[dofset_name];  // dofset id for scalar field
}

// TODO: shift to DRT::Utils...
/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
bool XFEM::LevelSetCoupling::HaveMatchingNodes(
    const Teuchos::RCP<DRT::Discretization>& dis_A, const Teuchos::RCP<DRT::Discretization>& dis_B)
{
  // check for equal node row maps
  const Epetra_Map* noderowmap_A = dis_A->NodeRowMap();
  const Epetra_Map* noderowmap_B = dis_B->NodeRowMap();

  if (!(noderowmap_A->SameAs(*noderowmap_B))) return false;

  // check for equal node coordinates
  for (int lid = 0; lid < noderowmap_A->NumMyElements(); ++lid)
  {
    const DRT::Node* node_A = dis_A->lRowNode(lid);
    const DRT::Node* node_B = dis_B->lRowNode(lid);

    const int nsd = node_A->Dim();

    CORE::LINALG::SerialDenseVector X_A(nsd);
    CORE::LINALG::SerialDenseVector X_B(nsd);

    std::copy(node_A->X().data(), node_A->X().data() + nsd, X_A.values());
    std::copy(node_B->X().data(), node_B->X().data() + nsd, X_B.values());

    CORE::LINALG::SerialDenseVector diff(X_A);
    diff.scale(-1.0);
    diff += X_B;

    if (CORE::LINALG::Norm2(diff) > 1e-14) return false;
  }

  return true;
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::InitStateVectors()
{
  // initialize state vectors w.r.t. background discretization
  InitStateVectors_Bg();
  // initialize state vectors w.r.t. cutter (potential subset of scatra) discretization
  init_state_vectors_cutter();
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::InitStateVectors_Bg()
{
  // background-dis (fluid) related state vectors
  const Epetra_Map* bg_dofrowmap = bg_dis_->dof_row_map(bg_nds_phi_);

  phinp_ = CORE::LINALG::CreateVector(*bg_dofrowmap, true);
}


// TODO: check if all vectors are really used and implement a save export strategy... (also used in
// mesh coupling?)
/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::init_state_vectors_cutter()
{
  // cutter-dis related state vectors
  const Epetra_Map* cutter_dofrowmap =
      cutter_dis_->dof_row_map(cutter_nds_phi_);  // used for level set field and its derivatives
  const Epetra_Map* cutter_dofcolmap =
      cutter_dis_->DofColMap(cutter_nds_phi_);  // used for level set field and its derivatives

  cutter_phinp_ = CORE::LINALG::CreateVector(*cutter_dofrowmap, true);
  cutter_phinp_col_ = CORE::LINALG::CreateVector(*cutter_dofcolmap, true);
  gradphinp_smoothed_node_ = CORE::LINALG::CreateMultiVector(*cutter_dofrowmap, nsd_, true);
  gradphinp_smoothed_node_col_ = CORE::LINALG::CreateMultiVector(*cutter_dofcolmap, nsd_, true);
  curvaturenp_node_ = CORE::LINALG::CreateVector(*cutter_dofrowmap, true);
  curvaturenp_node_col_ = CORE::LINALG::CreateVector(*cutter_dofcolmap, true);
}


// TODO: check output functionality
/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::PrepareCutterOutput()
{
  // -------------------------------------------------------------------
  // prepare output
  // -------------------------------------------------------------------
  cutter_output_ = cutter_dis_->Writer();

  if (cutter_output_ == Teuchos::null)
  {
    cutter_dis_->SetWriter(Teuchos::rcp(
        new IO::DiscretizationWriter(cutter_dis_, GLOBAL::Problem::Instance()->OutputControlFile(),
            GLOBAL::Problem::Instance()->spatial_approximation_type())));
  }

  bg_output_ = bg_dis_->Writer();
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::do_condition_specific_setup()
{
  // TODO: remove the smoothed normal stuff from this function!!!
  // read initial level-set field
  SetLevelSetField(time_);

  // set level-boolean type (may be overwritten in constructors of derived class
  set_level_set_boolean_type();
}

// TODO: store the first element condition to access cutterele_conds_[0] at several places
/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::set_level_set_boolean_type()
{
  if (cutterele_conds_.size() == 0)
    FOUR_C_THROW(
        "no element condition for LevelSetCouplingBC set. Not possible to extract BOOLEANTYPE!");

  CORE::Conditions::Condition* cond = (cutterele_conds_[0]).second;
  const std::string& booleantype = cond->parameters().Get<std::string>("booleantype");

  if (booleantype == "none")
    ls_boolean_type_ = ls_none;
  else if (booleantype == "cut")
    ls_boolean_type_ = ls_cut;
  else if (booleantype == "union")
    ls_boolean_type_ = ls_union;
  else if (booleantype == "difference")
    ls_boolean_type_ = ls_difference;
  else if (booleantype == "sym_difference")
    ls_boolean_type_ = ls_sym_difference;
  else
    FOUR_C_THROW("not a valid boolean type %s: ", booleantype.c_str());
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
bool XFEM::LevelSetCoupling::apply_complementary_operator()
{
  if (cutterele_conds_.size() == 0)
    FOUR_C_THROW(
        "no element condition for LevelSetCouplingBC set. Not possible to extract BOOLEANTYPE!");

  CORE::Conditions::Condition* cond = (cutterele_conds_[0]).second;
  bool complementary = (bool)cond->parameters().Get<int>("complementary");

  return complementary;
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::Output(
    const int step, const double time, const bool write_restart_data, const int lsc_idx)
{
  // output for level-set interface
  // bg_output_->NewStep(step,time); // not required, as already called for the bgdis when output is
  // written for fluid fields

  std::ostringstream temp;
  temp << lsc_idx;
  std::string name = "phinp_" + temp.str();

  bg_output_->WriteVector(name, phinp_);

  // write restart
  if (write_restart_data)
  {
    std::string name_restart = "phinp_res_" + temp.str();

    bg_output_->WriteVector(name_restart, phinp_);
  }

  cutter_output_->NewStep(step,
      time);  // required, as already called for the bgdis when output is written for fluid fields

  // write restart
  if (write_restart_data)
  {
    std::ostringstream temp2;
    temp2 << lsc_idx;
    std::string name_restart = "cutter_phinp_res_" + temp.str();

    cutter_output_->WriteVector(name_restart, cutter_phinp_);
  }
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::GmshOutput(const std::string& filename_base, const int step,
    const int gmsh_step_diff, const bool gmsh_debug_out_screen)
{
  // TODO: adapt!!!
  // gmsh output of geometric quantities based on the cutterdis (=fluid dis) (mapped quantities from
  // scatradis to fluiddis) cutterdis=fluiddis within the condition manager

  std::ostringstream filename_base_fsi;
  filename_base_fsi << filename_base << "_levelset";

  const std::string filename = IO::GMSH::GetNewFileNameAndDeleteOldFiles(filename_base_fsi.str(),
      cutter_output_->Output()->FileName(), step, gmsh_step_diff, gmsh_debug_out_screen, myrank_);

  std::ofstream gmshfilecontent(filename.c_str());

  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "SOLcutter-phi \" {" << std::endl;
    // draw vector field 'force' for every node
    IO::GMSH::ScalarFieldDofBasedToGmsh(
        cutter_dis_, cutter_phinp_, cutter_nds_phi_, gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }

  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "SOLcutter-smoothedgradphi \" {" << std::endl;
    IO::GMSH::VectorFieldMultiVectorDofBasedToGmsh(
        cutter_dis_, gradphinp_smoothed_node_, gmshfilecontent, cutter_nds_phi_);
    gmshfilecontent << "};" << std::endl;
  }

  // gmsh output for geometric quantities
}

// -------------------------------------------------------------------
// Read Restart data for cutter discretization
// -------------------------------------------------------------------
void XFEM::LevelSetCoupling::read_restart(const int step, const int lsc_idx)
{
  //  FOUR_C_THROW("Not tested Level Set restart from file. Should most likely work though if this
  //  FOUR_C_THROW is removed.");

  //-------- boundary discretization
  IO::DiscretizationReader boundaryreader(
      cutter_dis_, GLOBAL::Problem::Instance()->InputControlFile(), step);

  const double time = boundaryreader.ReadDouble("time");

  if (myrank_ == 0)
  {
    IO::cout << "            RESTART IS PERFORMED FROM FUNCTION IN INPUT FILE!                  "
             << IO::endl;
    IO::cout << "read_restart for Level Set Cut in Xfluid (time=" << time << " ; step=" << step
             << ")" << IO::endl;
  }

  SetLevelSetField(time);
}

// TODO: remove the Navier-Slip stuff
/*---------------------------------------------------------------------------*
 | Set the level set field and if smoothed gradients are needed create these |
 |                                                                           |
 *---------------------------------------------------------------------------*/
bool XFEM::LevelSetCoupling::SetLevelSetField(const double time)
{
  // TODO: clean this routine!!!

  // make a copy of last time step

  Teuchos::RCP<Epetra_Vector> delta_phi = CORE::LINALG::CreateVector(cutter_phinp_->Map(), true);
  delta_phi->Update(1.0, *cutter_phinp_, 0.0);

  // initializations
  int err(0);
  double value(0.0);
  std::vector<int> nodedofs;

  // get the function from the first element
  const int lid = 0;
  CORE::Conditions::Condition* cond = cutterele_conds_[lid].second;
  const int func_no = cond->parameters().Get<int>("levelsetfieldno");

  // loop all nodes on the processor
  for (int lnodeid = 0; lnodeid < cutter_dis_->NumMyRowNodes(); lnodeid++)
  {
    // get the processor's local scatra node
    DRT::Node* lnode = cutter_dis_->lRowNode(lnodeid);

    // get value
    if (func_no < 0)
      value = funct_implementation(func_no, lnode->X().data(), time);
    else if (func_no >= 1)
    {
      value = GLOBAL::Problem::Instance()
                  ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(func_no - 1)
                  .Evaluate(lnode->X().data(), time, 0);
    }
    else
      FOUR_C_THROW("invalid function no. to set level-set field!");

    const std::vector<int> lm = cutter_dis_->Dof(cutter_nds_phi_, lnode);

    if (lm.size() != 1) FOUR_C_THROW("assume 1 dof in cutterdis-Dofset for phi vector");

    const int gid = lm[0];
    const int lid = cutter_phinp_->Map().LID(gid);
    err = cutter_phinp_->ReplaceMyValues(1, &value, &lid);
    if (err) FOUR_C_THROW("could not replace values for phi vector");
  }


  // TODO: remove this part from this function!!!

  // Might make this available for other condition types!
  const INPAR::XFEM::EleCouplingCondType cond_type = CondType_stringToEnum(cond_name_);

  if (cond_type == INPAR::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP)
  {
    // Do we need smoothed gradients? I.e. what type is it?
    const int val = cutterele_conds_[lid].second->parameters().Get<int>("SURFACE_PROJECTION");
    projtosurf_ = static_cast<INPAR::XFEM::ProjToSurface>(val);

    if (projtosurf_ != INPAR::XFEM::Proj_normal)  // and projtosurf_!=INPAR::XFEM::Proj_normal_phi
    {
      // check for potential L2_Projection smoothing
      const int l2_proj_num = (cond->parameters().Get<int>("l2projsolv") + 1);
      if (l2_proj_num < 1) FOUR_C_THROW("Issue with L2_PROJECTION_SOLVER, smaller than 1!!!");

      // SMOOTHED GRAD PHI!!!!!! (Create from nodal map on Xfluid discretization)
      // This method might be a bit too complicated as we need to save modphinp (size of fluid
      // discretization), as well as gradphinp_smoothed_rownode as the function
      // compute_nodal_l2_projection returns the row node map.
      //----------------------------------------------
      // create the parameters for the discretization
      Teuchos::ParameterList eleparams;
      // action for elements
      eleparams.set<int>("action", FLD::presgradient_projection);

      // To get phi nodal values into pressure dofs in the fluid discretization!!! - any idea for
      // nice implementation?
      const Epetra_Map* modphinp_dofrowmap =
          Teuchos::rcp_dynamic_cast<DRT::DiscretizationXFEM>(cutter_dis_)->InitialDofRowMap();
      Teuchos::RCP<Epetra_Vector> modphinp =
          Teuchos::rcp(new Epetra_Vector(*modphinp_dofrowmap, true));

      double* val = cutter_phinp_->Values();

      int numrows = cutter_dis_->NumMyRowNodes();
      // loop all column nodes on the processor
      for (int lnodeid = 0; lnodeid < numrows; ++lnodeid)
      {
        // get the processor's local node
        DRT::Node* lsnode = cutter_dis_->lRowNode(lnodeid);
        if (lsnode == nullptr) FOUR_C_THROW("Returned node is null-pointer.");

        std::vector<int> initialdof =
            Teuchos::rcp_dynamic_cast<DRT::DiscretizationXFEM>(cutter_dis_)->InitialDof(lsnode);

        if (initialdof.size() != 4)
          FOUR_C_THROW("Initial Dof Size is not 4! Size: %d", initialdof.size());

        const int gid = initialdof[3];

        int err = modphinp->ReplaceGlobalValues(1, &val[lnodeid], &gid);
        if (err != 0) FOUR_C_THROW("Something went wrong when replacing the values.");

      }  // Loop over all nodes

      // SAFETY check
      // dependent on the desired projection, just remove this line
      if (not modphinp->Map().SameAs(
              *Teuchos::rcp_dynamic_cast<DRT::DiscretizationXFEM>(cutter_dis_)->InitialDofRowMap()))
        FOUR_C_THROW("input map is not a dof row map of the fluid");

      // set given state for element evaluation
      cutter_dis_->ClearState();
      Teuchos::rcp_dynamic_cast<DRT::DiscretizationXFEM>(cutter_dis_)
          ->SetInitialState(0, "pres", modphinp);

      // Lives on NodeRow-map!!!
      const auto& solverparams = GLOBAL::Problem::Instance()->SolverParams(l2_proj_num);
      Teuchos::RCP<Epetra_MultiVector> gradphinp_smoothed_rownode =
          CORE::FE::compute_nodal_l2_projection(cutter_dis_, "pres", 3, eleparams, solverparams);
      if (gradphinp_smoothed_rownode == Teuchos::null)
        FOUR_C_THROW("A smoothed grad phi is required, but an empty one is provided!");

      // The following bugfix needs to be check carefully
      {
        // Convert NodeRowMap from compute_nodal_l2_projection to dof_row_map while assuming
        // identical ordering
        for (int ivec = 0; ivec < gradphinp_smoothed_rownode->NumVectors(); ivec++)
        {
          Epetra_Vector* itemp = (*gradphinp_smoothed_rownode)(ivec);
          for (int jlength = 0; jlength < itemp->MyLength(); jlength++)
          {
            gradphinp_smoothed_node_->ReplaceMyValue(jlength, ivec, itemp->operator[](jlength));
          }
        }
        // Bring dof_row_map to DofColMap layout (Attention: name is node but lives on dof)
        CORE::LINALG::Export(*gradphinp_smoothed_node_, *gradphinp_smoothed_node_col_);
      }

      //---------------------------------------------- // SMOOTHED GRAD PHI END
    }
  }

  // map the cutterdis-based phinp to the bgdis-noderowmap based phinp
  MapCutterToBgVector(cutter_dis_, cutter_phinp_, cutter_nds_phi_, bg_dis_, phinp_, bg_nds_phi_);

  // check if boundary position changed from the last step

  delta_phi->Update(1.0, *cutter_phinp_, -1.0);  // phinp - phin

  double norm = 0.0;
  delta_phi->Norm2(&norm);

  return (norm > 1e-14);  // did interface change?
}


// TODO: generalization in DRT::UTILS???
/*---------------------------------------------------------------------------*
 *---------------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::MapCutterToBgVector(
    const Teuchos::RCP<DRT::Discretization>& source_dis,
    const Teuchos::RCP<Epetra_Vector>& source_vec_dofbased, const int source_nds,
    const Teuchos::RCP<DRT::Discretization>& target_dis,
    const Teuchos::RCP<Epetra_Vector>& target_vec_dofbased, const int target_nds)
{
  if (HaveMatchingNodes(source_dis, target_dis))  // check for equal node positions
  {
    // here we assume that source_dis and target_dis are equal!

    // loop the nodes
    for (int lnodeid = 0; lnodeid < target_dis->NumMyRowNodes(); ++lnodeid)
    {
      DRT::Node* node_source = source_dis->lRowNode(lnodeid);
      DRT::Node* node_target = target_dis->lRowNode(lnodeid);

      // get the set of source dof IDs for this node
      std::vector<int> lm_source;
      source_dis->Dof(source_nds, node_source, lm_source);

      std::vector<int> lm_target;
      target_dis->Dof(target_nds, node_target, lm_target);

      if (static_cast<int>(lm_source.size()) != 1)
        FOUR_C_THROW("we expect a unique dof per node here!");

      if (static_cast<int>(lm_target.size()) != 1)
        FOUR_C_THROW("we expect a unique dof per node here!");

      std::vector<double> val_source;
      CORE::FE::ExtractMyValues(*source_vec_dofbased, val_source, lm_source);

      // set to a dofrowmap based vector!
      const int lid_target = target_vec_dofbased->Map().LID(lm_target[0]);
      const int err = target_vec_dofbased->ReplaceMyValues(1, val_source.data(), &lid_target);
      if (err) FOUR_C_THROW("could not replace values for convective velocity");
    }
  }
  else
  {
    FOUR_C_THROW("nonmatching discretizations not supported so far! - Implement a mesh projector?");
  }
}

Teuchos::RCP<Epetra_Vector> XFEM::LevelSetCoupling::get_level_set_field_as_node_row_vector()
{
  Teuchos::RCP<Epetra_Vector> bg_phinp_nodemap_ =
      CORE::LINALG::CreateVector(*bg_dis_->NodeRowMap(), true);

  // loop the nodes
  for (int lnodeid = 0; lnodeid < bg_dis_->NumMyRowNodes(); ++lnodeid)
  {
    DRT::Node* node = bg_dis_->lRowNode(lnodeid);
    std::vector<int> lm_source;
    bg_dis_->Dof(bg_nds_phi_, node, lm_source);

    std::vector<double> val_source;
    CORE::FE::ExtractMyValues(*phinp_, val_source, lm_source);

    if (val_source.size() != 1) FOUR_C_THROW("we expect only one dof");

    const int lid_target = bg_dis_->NodeRowMap()->LID(node->Id());
    const int err = bg_phinp_nodemap_->ReplaceMyValues(1, val_source.data(), &lid_target);
    if (err) FOUR_C_THROW("could not replace values for phi vector");
  }

  return bg_phinp_nodemap_;
}



/*----------------------------------------------------------------------*
 | set interface level set field at current time           schott 02/15 |
 *----------------------------------------------------------------------*/
double XFEM::LevelSetCoupling::funct_implementation(
    const int func_no, const double* coords, const double t)
{
  //  FOUR_C_THROW("you try to evaluate an implemented function for level-set field! Which one?");
  // WARNING!

  double x = coords[0];
  double y = coords[1];
  double z = coords[2];

  double val = 0.0;

  double R = 0.2;
  double r = 0.1;

  const double alpha = 0.6;


  if (func_no == -1)
  {
    // level set field for a helical pipe
    // sqrt( (x-Rcos(2 pi t))^2 + (y-Rsin(2 pi t))^2 + (z-alpha t)^2 -r = 0
    // with t(x,y,z) solution of minimization problem
    // dist((x,y,z), curve(t(x,y,z))) = min!

    // with curve(t) a parametrized curve (e.g. a helix)



    // NEWTON SYSTEM FOR SOLVING FOR t
    // d''(t) Delta_t = -d'(t)
    // t_n+1 = t_n + Delta_t

    // HELICAL CURVE z=alpha*t


    const double two_alpha_squared = 2.0 * alpha * alpha;
    double two_PI = 2.0 * M_PI;

    double t_0 = z / alpha;

    double Jac = 0.0;
    double rhs = 1.0;

    int armijo_steps = 50;

    int maxiter = 50;

    for (int i = 0; i < maxiter; i++)
    {
      if (fabs(rhs) < 1e-13) break;

      double arc = two_PI * t_0;
      double cosine = cos(arc);
      double sine = sin(arc);
      Jac = 4.0 * M_PI * R * (two_PI * x * cosine + two_PI * y * sine) + two_alpha_squared;
      rhs = 4.0 * M_PI * R * (x * sine - y * cosine) + two_alpha_squared * t_0 - 2.0 * alpha * z;


      double dt = -rhs / Jac;


      double armijo = 1.0;

      if (i < armijo_steps)
      {
        // it may happen, that the Newton direction is not a descent direction, then change the
        // search direction grad(f(x))^T * searchdir < 0 !   <=>  d'(t)*dt < 0   <=>  rhs*dt < 0
        if (dt * rhs > 0.0) dt *= -1.0;

        for (int l = 0; l < 5; ++l)
        {
          if (l > 0) armijo *= 0.5;

          // d(t+armijo*dt) < d(t) !!! and armijo (0,1] moeglichst nahe an 1
          double t_new = t_0 + armijo * dt;
          double arc_new = two_PI * t_new;
          double cosine_new = cos(arc_new);
          double sine_new = sin(arc_new);

          double tmpx_new = x - R * cosine_new;
          double tmpy_new = y - R * sine_new;
          double tmpz_new = z - alpha * t_new;
          double norm1_squared = tmpx_new * tmpx_new + tmpy_new * tmpy_new + tmpz_new * tmpz_new;

          double tmpx = x - R * cosine;
          double tmpy = y - R * sine;
          double tmpz = z - alpha * t_0;
          double norm2_squared = tmpx * tmpx + tmpy * tmpy + tmpz * tmpz;

          if (norm1_squared < norm2_squared) break;
        }
      }

      t_0 += dt * armijo;

      if (i > maxiter - 1)
      {
        std::cout << "Jac: " << Jac << std::endl;
        std::cout << "i: " << i << " rhs " << rhs << std::endl;
        std::cout << "armijo: " << armijo << std::endl;

        FOUR_C_THROW(
            "did not converge properly, intial guess not good enough - increase helixal height "
            "alpha!");
      }
    }


    double curve = alpha * t_0;

    double angle = two_PI * t_0;

    double cosine = cos(angle);
    double sine = sin(angle);

    double tmp1 = x - R * cosine;
    double tmp2 = y - R * sine;
    double tmp3 = z - curve;

    double val_helix = sqrt(tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3) - r;

    return val_helix;
  }
  else if (func_no == -2)
  {
    double n1 = 0.0;
    double n2 = 2.0 * M_PI * R;
    double n3 = alpha;

    double norm = sqrt(n1 * n1 + n2 * n2 + n3 * n3);
    n1 /= norm;
    n2 /= norm;
    n3 /= norm;

    // inflow region
    // point_on_plane_x
    double pop_x = 0.0;  // here arbitrary
    double pop_y = 0.0;
    double pop_z = -2.0 * alpha;

    double dist = n1 * pop_x + n2 * pop_y + n3 * pop_z;

    double val_plane_inflow = n1 * x + n2 * y + n3 * z - dist;

    double val_inflow = std::max(val_plane_inflow, z - (pop_z + r * 1.1));
    val_inflow = std::min(val_inflow, (z - (pop_z - r * 1.1)));

    return -val_inflow;
  }
  else if (func_no == -3)
  {
    // outflow region

    double n1_out = 0.0;
    double n2_out = -2.0 * M_PI * R;
    double n3_out = -alpha;

    double norm_out = sqrt(n1_out * n1_out + n2_out * n2_out + n3_out * n3_out);
    n1_out /= norm_out;
    n2_out /= norm_out;
    n3_out /= norm_out;

    // point_on_plane_x
    double pop_x_out = 0.0;  // here arbitrary
    double pop_y_out = 0.0;
    double pop_z_out = +2.0 * alpha;

    double dist_out = n1_out * pop_x_out + n2_out * pop_y_out + n3_out * pop_z_out;

    double val_plane_outflow = n1_out * x + n2_out * y + n3_out * z - dist_out;

    double val_outflow = std::max(val_plane_outflow, -(z - (pop_z_out - r * 1.1)));
    val_outflow = std::min(val_outflow, -(z - (pop_z_out + r * 1.1)));

    return -val_outflow;
  }
  else if (func_no == -4)
  {
    double val_inner_ring_cyl = sqrt((x + 0.2) * (x + 0.2) + y * y) - 0.14;
    double val_z_limit_inner = z + 0.9;
    return std::max(val_inner_ring_cyl, val_z_limit_inner);
  }
  else if (func_no == -5)
  {
    double val_outer_ring_cyl = sqrt((x + 0.2) * (x + 0.2) + y * y) - 0.22;
    double val_inner_ring_cyl = sqrt((x + 0.2) * (x + 0.2) + y * y) - 0.14;
    double val_ring = std::max(val_outer_ring_cyl, -val_inner_ring_cyl);
    double val_z_limit_inner = z + 0.9;
    double val_cylinder_ring_half = std::max(val_ring, val_z_limit_inner);
    return val_cylinder_ring_half;
  }
  else if (func_no == -6)  // cylinder at inflow of a helix
  {
    double n1 = 0.0;
    double n2 = 2.0 * M_PI * R;
    double n3 = alpha;

    double norm = sqrt(n1 * n1 + n2 * n2 + n3 * n3);

    n1 /= norm;
    n2 /= norm;
    n3 /= norm;

    // inflow region
    // point_on_plane_x
    double pop_x = 0.2;  // here arbitrary
    double pop_y = 0.0;
    double pop_z = -2.0 * alpha;

    double dist = n1 * pop_x + n2 * pop_y + n3 * pop_z;

    double val_plane_inflow = n1 * x + n2 * y + n3 * z - dist;

    double coord_x_center = x - 0.2;
    double coord_y_center = y;
    double coord_z_center = z + alpha * 2.0;

    double coord_dot_n = coord_x_center * n1 + coord_y_center * n2 + coord_z_center * n3;

    double tmp1 = (coord_x_center - n1 * coord_dot_n);
    tmp1 *= tmp1;
    double tmp2 = (coord_y_center - n2 * coord_dot_n);
    tmp2 *= tmp2;
    double tmp3 = (coord_z_center - n3 * coord_dot_n);
    tmp3 *= tmp3;

    double val_cylinder = sqrt(tmp1 + tmp2 + tmp3) - r;
    val_cylinder = std::max(val_cylinder, val_plane_inflow);

    return val_cylinder;
  }
  else if (func_no == -7)  // box for Oseen
  {
    // return -(std::max( (fabs(x-0.5+0.013))/0.3, (fabs(y-0.5+0.013))/0.3)-1.0);
    return -(
        std::max((fabs(x - 1.0)) / 0.45, std::max((fabs(y - 0.5)) / 0.45, (fabs(z - 0.5)) / 0.45)) -
        1.0);
  }


  // val = std::max(val_helix, std::max(val_inflow, val_outflow) );


  ////  double z_centerline = 1.0;
  //  double alpha = 0.21;
  //
  //  double arc=0.0;
  //  int n=0;
  //
  //  double z_max_at_arczero = r;
  //
  ////  if(fabs(x)<1e-14 or fabs(y)<1e-14)
  //////    if(fabs(x)<1e-14 and fabs(y)<1e-14)
  ////  {
  ////    val = 0.5;
  ////    return val;
  ////  }
  //
  //  double sgn_y = 1.0;
  //
  //  if(y>1e-14)
  //    sgn_y= 1.0;
  //  else if(y<1e-14)
  //    sgn_y= -1.0;
  //  else
  //    sgn_y = 0.0;
  //
  //  double sgn_x = 1.0;
  //
  //  if(x>1e-14)
  //    sgn_x= 1.0;
  //  else if(x<1e-14)
  //    sgn_x= -1.0;
  //  else
  //    sgn_x = 0.0;
  //
  //  n = 0;
  //
  //  if(z>=0.0)
  //  {
  //    // look for the first
  //    for(int i = 0; ; i++)
  //    {
  //      if(i*alpha <= z and z<(i+1)*alpha)
  //      {
  //        n=i;
  //        break;
  //      }
  //    }
  //  }
  //  else // z<0.0
  //  {
  //    // look for the first
  //    for(int i = 0; ; i--)
  //    {
  //      if(i*alpha <= z and z<(i+1)*alpha)
  //      {
  //        n=i;
  //        break;
  //      }
  //    }
  //  }
  //
  //  // three possible i's
  ////  if(fabs(z-(n-1)*alpha) < fabs(z-n*alpha))
  ////    n--;
  //
  //  double arc_1 = 0.0;
  //  double arc_2 = 0.0;
  //  double arc_3 = 0.0;
  //
  //
  //  if(fabs(x)>1e-14 and fabs(y)>1e-14)
  //  {
  //    arc_1 = sgn_y* 1.0/(2.0*PI)*acos(sgn_x/(sqrt(1+yy/xx))) + 0.5*(1.0 - sgn_y) +n;
  //    arc_2 = sgn_y* 1.0/(2.0*PI)*acos(sgn_x/(sqrt(1+yy/xx))) + 0.5*(1.0 - sgn_y) +(n-1);
  ////    arc_3 = sgn_y* 1.0/(2.0*PI)*acos(sgn_x/(sqrt(1+yy/xx))) + 0.5*(1.0 - sgn_y) +(n-2);
  //
  //    //arc_3 = sgn_y* 1.0/(2.0*PI)*acos(sgn_x/(sqrt(1+yy/xx))) + 0.5*(1.0 - sgn_y) +(n+1);
  //  }
  //  arc= std::max(arc_1,arc_2);
  // //   arc= std::max(arc_1,std::max(arc_2,arc_3));
  // //sgn_y* 1.0/(2.0*PI)*acos(sgn_x/(sqrt(1+yy/xx))) + 0.5*(1.0 - sgn_y) +n;
  ////  else
  ////  {
  ////    if(fabs(x)>1e-14 and fabs(y)<1e-14)
  ////    {
  ////      if(x>0)
  ////        arc = n;
  ////      else
  ////        arc = 0.5+n;
  ////    }
  ////    else if(fabs(x)<1e-14 and fabs(y)>1e-14)
  ////    {
  ////      if(y>0)
  ////        arc = 0.25+n;
  ////      else
  ////        arc = 0.75+n;
  ////    }
  ////    else
  ////    {
  ////      arc = 0.5 + n;
  ////    }
  ////  }
  //
  ////  if(y>1e-14)
  ////  {
  ////    arc= sgn* 1.0/(2.0*PI)*acos(1.0/(sqrt(1+yy/xx))) +n;
  ////  }
  ////  else if(y < 1e-14)
  ////  {
  ////    arc= 1.0-1.0/(2.0*PI)*acos(1.0/(sqrt(1+yy/xx))) +n-1;
  ////  }
  ////  else
  ////  {
  ////    if(x > 0.0)
  ////      arc=n;
  ////    else if(x< 0.0)
  ////      arc=0.5+n;
  ////    else
  ////      arc=0.0;
  ////
  ////  }
  //
  //  double z_centerline = alpha * arc;
  //
  //  double tmp_1 = sqrt(xx+yy)-R;
  //  double tmp_2 = z-z_centerline;
  //
  //  val = tmp_1*tmp_1 + tmp_2*tmp_2-r*r;

  return val;
}

// TODO: has_interface_moved_ checks its functionality and there is another flag in meshcoupling i
// think
XFEM::LevelSetCouplingBC::LevelSetCouplingBC(
    Teuchos::RCP<DRT::Discretization>& bg_dis,  ///< background discretization
    const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                   ///< discretization is identified
    Teuchos::RCP<DRT::Discretization>&
        cond_dis,           ///< full discretization from which the cutter discretization is derived
    const int coupling_id,  ///< id of composite of coupling conditions
    const double time,      ///< time
    const int step          ///< time step
    )
    : LevelSetCoupling(bg_dis, cond_name, cond_dis, coupling_id, time, step),
      has_interface_moved_(true)
{
}

/*----------------------------------------------------------------------*
 | set interface level set field at current time           schott 02/15 |
 *----------------------------------------------------------------------*/
void XFEM::LevelSetCouplingBC::PrepareSolve()
{
  if (myrank_ == 0) IO::cout << "\t set level-set field, time " << time_ << IO::endl;

  has_interface_moved_ = SetLevelSetField(time_);
  return;
}



bool XFEM::LevelSetCouplingBC::HasMovingInterface() { return has_interface_moved_; }



void XFEM::LevelSetCouplingWeakDirichlet::evaluate_coupling_conditions(
    CORE::LINALG::Matrix<3, 1>& ivel, CORE::LINALG::Matrix<3, 1>& itraction,
    const CORE::LINALG::Matrix<3, 1>& x, const CORE::Conditions::Condition* cond)
{
  // evaluate interface velocity (given by weak Dirichlet condition)
  evaluate_dirichlet_function(ivel, x, cond, time_);

  // no interface traction to be evaluated
  itraction.Clear();
}

// TODO: remove old state implementation?!
void XFEM::LevelSetCouplingWeakDirichlet::evaluate_coupling_conditions_old_state(
    CORE::LINALG::Matrix<3, 1>& ivel, CORE::LINALG::Matrix<3, 1>& itraction,
    const CORE::LINALG::Matrix<3, 1>& x, const CORE::Conditions::Condition* cond)
{
  // evaluate interface velocity (given by weak Dirichlet condition)
  evaluate_dirichlet_function(ivel, x, cond, time_ - dt_);

  // no interface traction to be evaluated
  itraction.Clear();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingWeakDirichlet::setup_configuration_map()
{
  // Configuration of Consistency Terms
  configuration_map_[INPAR::XFEM::F_Con_Row] = std::pair<bool, double>(true, 1.0);
  configuration_map_[INPAR::XFEM::F_Con_Col] = std::pair<bool, double>(true, 1.0);

  // Configuration of Adjount Consistency Terms
  configuration_map_[INPAR::XFEM::F_Adj_Row] = std::pair<bool, double>(true, 1.0);
  configuration_map_[INPAR::XFEM::F_Adj_Col] = std::pair<bool, double>(true, 1.0);

  // Configuration of Penalty Terms
  configuration_map_[INPAR::XFEM::F_Pen_Row] = std::pair<bool, double>(true, 1.0);
  configuration_map_[INPAR::XFEM::F_Pen_Col] = std::pair<bool, double>(true, 1.0);
  return;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingWeakDirichlet::update_configuration_map_gp(
    double& kappa_m,         //< fluid sided weighting
    double& visc_m,          //< master sided dynamic viscosity
    double& visc_s,          //< slave sided dynamic viscosity
    double& density_m,       //< master sided density
    double& visc_stab_tang,  //< viscous tangential NIT Penalty scaling
    double& full_stab, const CORE::LINALG::Matrix<3, 1>& x, const CORE::Conditions::Condition* cond,
    DRT::Element* ele,   //< Element
    DRT::Element* bele,  //< Boundary Element
    double* funct,       //< local shape function for Gauss Point (from fluid element)
    double* derxy,  //< local derivatives of shape function for Gauss Point (from fluid element)
    CORE::LINALG::Matrix<3, 1>& rst_slave,  //< local coord of gp on slave boundary element
    CORE::LINALG::Matrix<3, 1>& normal,     //< normal at gp
    CORE::LINALG::Matrix<3, 1>& vel_m,      //< master velocity at gp
    double* fulltraction                    //< precomputed fsi traction (sigmaF n + gamma relvel)
)
{
  // Configuration of Penalty Terms
  configuration_map_[INPAR::XFEM::F_Pen_Row].second = full_stab;

  return;
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNeumann::do_condition_specific_setup()
{
  // Call Base Class
  XFEM::LevelSetCouplingBC::do_condition_specific_setup();

  // Check if Inflow Stabilisation is active
  if (!cutterele_conds_.size()) FOUR_C_THROW("cutterele_conds_.size = 0!");
  CORE::Conditions::Condition* cond = (cutterele_conds_[0]).second;
  auto inflow_stab = cond->parameters().Get<bool>("InflowStab");
  for (auto& cutterele_cond : cutterele_conds_)
  {
    CORE::Conditions::Condition* cond = cutterele_cond.second;
    auto this_inflow = cond->parameters().Get<bool>("InflowStab");
    if (inflow_stab != this_inflow)
      FOUR_C_THROW(
          "You want to stabilized just some of your Neumann Boundaries? - feel free to implement!");
  }

  if (inflow_stab)
  {
    std::cout << "==| LevelSetCouplingNeumann: Inflow Stabilization active! |==" << std::endl;
    inflow_stab_ = true;
  }
  else
    inflow_stab_ = false;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNeumann::setup_configuration_map()
{
  if (inflow_stab_)
  {
    // Configuration of Penalty Terms
    configuration_map_[INPAR::XFEM::F_Pen_Col] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::F_Adj_Col] =
        std::pair<bool, double>(true, 1.0);  //<-- IMPORTANT!: used for the constraint scaling
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNeumann::update_configuration_map_gp(
    double& kappa_m,         //< fluid sided weighting
    double& visc_m,          //< master sided dynamic viscosity
    double& visc_s,          //< slave sided dynamic viscosity
    double& density_m,       //< master sided density
    double& visc_stab_tang,  //< viscous tangential NIT Penalty scaling
    double& full_stab, const CORE::LINALG::Matrix<3, 1>& x, const CORE::Conditions::Condition* cond,
    DRT::Element* ele,   //< Element
    DRT::Element* bele,  //< Boundary Element
    double* funct,       //< local shape function for Gauss Point (from fluid element)
    double* derxy,  //< local derivatives of shape function for Gauss Point (from fluid element)
    CORE::LINALG::Matrix<3, 1>& rst_slave,  //< local coord of gp on slave boundary element
    CORE::LINALG::Matrix<3, 1>& normal,     //< normal at gp
    CORE::LINALG::Matrix<3, 1>& vel_m,      //< master velocity at gp
    double* fulltraction                    //< precomputed fsi traction (sigmaF n + gamma relvel)
)
{
  if (inflow_stab_)
  {
    // Configuration of Penalty Terms
    double veln = normal.Dot(vel_m);  // as the normal is the structural body, inflow is positive
    if (veln < 0)
    {
      configuration_map_[INPAR::XFEM::F_Pen_Row] = std::pair<bool, double>(true, -density_m * veln);
      configuration_map_[INPAR::XFEM::F_Pen_Row_linF1] =
          std::pair<bool, double>(true, -density_m * normal(0));
      configuration_map_[INPAR::XFEM::F_Pen_Row_linF2] =
          std::pair<bool, double>(true, -density_m * normal(1));
      configuration_map_[INPAR::XFEM::F_Pen_Row_linF3] =
          std::pair<bool, double>(true, -density_m * normal(2));
    }
    else
    {
      configuration_map_[INPAR::XFEM::F_Pen_Row] = std::pair<bool, double>(false, 0);
      configuration_map_[INPAR::XFEM::F_Pen_Row_linF1] = std::pair<bool, double>(false, 0);
      configuration_map_[INPAR::XFEM::F_Pen_Row_linF2] = std::pair<bool, double>(false, 0);
      configuration_map_[INPAR::XFEM::F_Pen_Row_linF3] = std::pair<bool, double>(false, 0);
    }
  }

  return;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNeumann::evaluate_coupling_conditions(CORE::LINALG::Matrix<3, 1>& ivel,
    CORE::LINALG::Matrix<3, 1>& itraction, const CORE::LINALG::Matrix<3, 1>& x,
    const CORE::Conditions::Condition* cond)
{
  // no interface velocity to be evaluated
  ivel.Clear();

  // evaluate interface traction (given by Neumann condition)
  evaluate_neumann_function(itraction, x, cond, time_);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNeumann::evaluate_coupling_conditions(CORE::LINALG::Matrix<3, 1>& ivel,
    CORE::LINALG::Matrix<6, 1>& itraction, const CORE::LINALG::Matrix<3, 1>& x,
    const CORE::Conditions::Condition* cond)
{
  // no interface velocity to be evaluated
  ivel.Clear();

  // evaluate interface traction (given by Neumann condition)
  evaluate_neumann_function(itraction, x, cond, time_);
}

// TODO: combine it with the function before with optional time parameter?!
/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNeumann::evaluate_coupling_conditions_old_state(
    CORE::LINALG::Matrix<3, 1>& ivel, CORE::LINALG::Matrix<3, 1>& itraction,
    const CORE::LINALG::Matrix<3, 1>& x, const CORE::Conditions::Condition* cond)
{
  // no interface velocity to be evaluated
  ivel.Clear();

  // evaluate interface traction (given by Neumann condition)
  evaluate_neumann_function(itraction, x, cond, time_ - dt_);
}

// TODO: inheritance from the bc seems to be the wrong concept, more delegate functionality to a
// Dirichlet object and a Neumann object... the same is implemented in mesh coupling object again!
// wrong concept!
/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
XFEM::LevelSetCouplingNavierSlip::LevelSetCouplingNavierSlip(
    Teuchos::RCP<DRT::Discretization>& bg_dis,  ///< background discretization
    const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                   ///< discretization is identified
    Teuchos::RCP<DRT::Discretization>&
        cond_dis,           ///< full discretization from which the cutter discretization is derived
    const int coupling_id,  ///< id of composite of coupling conditions
    const double time,      ///< time
    const int step          ///< time step
    )
    : LevelSetCouplingBC(bg_dis, cond_name, cond_dis, coupling_id, time, step)
{
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNavierSlip::set_element_conditions()
{
  XFEM::LevelSetCouplingBC::set_element_conditions();

  if (cutterele_conds_.size() == 0) FOUR_C_THROW("call set_element_conditions() first!");

  CORE::Conditions::Condition* cond = cutterele_conds_[0].second;  // get condition of first element

  // Get robin coupling IDs
  robin_dirichlet_id_ = cond->parameters().Get<int>("robin_id_dirch");
  robin_neumann_id_ = cond->parameters().Get<int>("robin_id_neumann");

  has_neumann_jump_ = (robin_neumann_id_ < 0) ? false : true;

  if (has_neumann_jump_)
  {
    std::cout << "#################################################################################"
                 "########################\n";
    std::cout << "#################################################################################"
                 "########################\n";
    std::cout << "### WARNING:: XFEM::LevelSetCouplingNavierSlip                              The "
                 "traction jump is      ###\n";
    std::cout << "### divided by the dynviscosity on Gausspoint Level, this might be expensed and "
                 "not really necessary! ###\n";
    std::cout << "#################################################################################"
                 "########################\n";
    std::cout << "#################################################################################"
                 "########################"
              << std::endl;
  }

  // set the navier-slip specific element conditions
  set_element_specific_conditions(
      cutterele_cond_robin_dirichlet_, "XFEMRobinDirichletVol", robin_dirichlet_id_);
  if (has_neumann_jump_)
    set_element_specific_conditions(
        cutterele_cond_robin_neumann_, "XFEMRobinNeumannVol", robin_neumann_id_);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNavierSlip::set_element_specific_conditions(
    std::vector<CORE::Conditions::Condition*>& cutterele_cond, const std::string& cond_name,
    const int& robin_id)
{
  // TODO: can we combine this function with set_element_conditions in the coupling base routine!

  // number of column cutter boundary elements
  int nummycolele = cutter_dis_->NumMyColElements();

  cutterele_cond.clear();
  cutterele_cond.reserve(nummycolele);

  //// initialize the vector invalid coupling-condition type "NONE"
  // EleCoupCond init_pair = EleCoupCond(INPAR::XFEM::CouplingCond_NONE,nullptr);
  for (int lid = 0; lid < nummycolele; ++lid) cutterele_cond.push_back(nullptr);

  //-----------------------------------------------------------------------------------
  // loop all column cutting elements on this processor
  for (int lid = 0; lid < nummycolele; ++lid)
  {
    DRT::Element* cutele = cutter_dis_->lColElement(lid);

    // get all conditions with given condition name
    std::vector<CORE::Conditions::Condition*> mycond;
    CORE::Conditions::FindElementConditions(cutele, cond_name, mycond);

    std::vector<CORE::Conditions::Condition*> mynewcond;
    get_condition_by_robin_id(mycond, robin_id, mynewcond);

    CORE::Conditions::Condition* cond_unique = nullptr;

    // safety checks
    if (mynewcond.size() != 1)
    {
      FOUR_C_THROW(
          "%i conditions of the same name with robin id %i, for element %i! %s coupling-condition "
          "not unique!",
          mynewcond.size(), (robin_id + 1), cutele->Id(), cond_name.c_str());
    }
    else if (mynewcond.size() == 1)  // unique condition found
    {
      cond_unique = mynewcond[0];
    }

    // store the unique condition pointer to the cutting element
    cutterele_cond[lid] = cond_unique;
  }
  //  //-----------------------------------------------------------------------------------
  //  // check if all column cutter elements have a valid condition type
  //  // loop all column cutting elements on this processor
  for (int lid = 0; lid < nummycolele; ++lid)
  {
    if (cutterele_cond[lid] == nullptr)
      FOUR_C_THROW("cutter element with local id %i has no Robin-condition!!!", lid);
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNavierSlip::evaluate_coupling_conditions(
    CORE::LINALG::Matrix<3, 1>& ivel, CORE::LINALG::Matrix<3, 1>& itraction,
    const CORE::LINALG::Matrix<3, 1>& x, const CORE::Conditions::Condition* cond)
{
  if (cutterele_cond_robin_dirichlet_.size() == 0)
    FOUR_C_THROW("initialize cutterele_cond_robin_dirichlet_ first!");

  // evaluate interface velocity (given by weak Dirichlet condition)
  evaluate_dirichlet_function(ivel, x, cutterele_cond_robin_dirichlet_[0], time_);

  // evaluate interface traction (given by Neumann condition)
  if (has_neumann_jump_)
  {
    if (cutterele_cond_robin_neumann_.size() == 0)
      FOUR_C_THROW("initialize cutterele_cond_robin_neumann_ first!");

    evaluate_neumann_function(itraction, x, cutterele_cond_robin_neumann_[0], time_);
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNavierSlip::evaluate_coupling_conditions_old_state(
    CORE::LINALG::Matrix<3, 1>& ivel, CORE::LINALG::Matrix<3, 1>& itraction,
    const CORE::LINALG::Matrix<3, 1>& x, const CORE::Conditions::Condition* cond)
{
  if (cutterele_cond_robin_dirichlet_.size() == 0)
    FOUR_C_THROW("initialize cutterele_cond_robin_dirichlet_ first!");

  // evaluate interface velocity (given by weak Dirichlet condition)
  evaluate_dirichlet_function(ivel, x, cutterele_cond_robin_dirichlet_[0], time_ - dt_);

  // evaluate interface traction (given by Neumann condition)
  if (has_neumann_jump_)
  {
    if (cutterele_cond_robin_neumann_.size() == 0)
      FOUR_C_THROW("initialize cutterele_cond_robin_neumann_ first!");

    evaluate_neumann_function(itraction, x, cutterele_cond_robin_neumann_[0], time_ - dt_);
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNavierSlip::GetSlipCoefficient(
    double& slipcoeff, const CORE::LINALG::Matrix<3, 1>& x, const CORE::Conditions::Condition* cond)
{
  if (is_constant_sliplength_)
    slipcoeff = sliplength_;
  else
    evaluate_scalar_function(slipcoeff, x.A(), sliplength_, cond, time_);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNavierSlip::set_condition_specific_parameters()
{
  if (cutterele_conds_.size() == 0) FOUR_C_THROW("call set_element_conditions() first!");

  CORE::Conditions::Condition* cond = cutterele_conds_[0].second;  // get condition of first element

  // Get the scaling factor for the slip length
  sliplength_ = cond->parameters().Get<double>("slipcoeff");

  // Temporary variable for readability.
  bool tmp_bool;

  // Is the slip length constant? Don't call functions at GP-level unnecessary.
  tmp_bool = (cond->parameters().Get<int>("funct") < 1);
  is_constant_sliplength_ = (tmp_bool) ? true : false;

  // Project the prescribed velocity in tangential direction, to remove "spurious velocities"
  //  from the geometry approximation.
  tmp_bool = ((cond->parameters().Get<int>("force_tang_vel")) == 0);
  forcetangvel_ = (tmp_bool) ? false : true;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNavierSlip::get_condition_by_robin_id(
    const std::vector<CORE::Conditions::Condition*>& mycond, const int coupling_id,
    std::vector<CORE::Conditions::Condition*>& mynewcond)
{
  mynewcond.clear();

  // select the conditions with specified "couplingID"
  for (size_t i = 0; i < mycond.size(); ++i)
  {
    CORE::Conditions::Condition* cond = mycond[i];
    const int id = cond->parameters().Get<int>("robin_id");

    if (id == coupling_id) mynewcond.push_back(cond);
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNavierSlip::setup_configuration_map()
{
  if (get_averaging_strategy() == INPAR::XFEM::Xfluid_Sided)
  {
    // Configuration of Consistency Terms
    configuration_map_[INPAR::XFEM::F_Con_Row] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::F_Con_Col] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::F_Con_t_Row] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::F_Con_t_Col] = std::pair<bool, double>(true, 1.0);

    // Configuration of Adjount Consistency Terms
    configuration_map_[INPAR::XFEM::F_Adj_n_Row] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::F_Adj_n_Col] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::F_Adj_t_Row] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::F_Adj_t_Col] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::FStr_Adj_t_Col] = std::pair<bool, double>(true, 1.0);

    // Configuration of Penalty Terms
    configuration_map_[INPAR::XFEM::F_Pen_n_Row] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::F_Pen_n_Col] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::F_Pen_t_Row] = std::pair<bool, double>(true, 1.0);
    configuration_map_[INPAR::XFEM::F_Pen_t_Col] = std::pair<bool, double>(true, 1.0);
  }
  else if (get_averaging_strategy() == INPAR::XFEM::invalid)
    FOUR_C_THROW("XFEM::LevelSetCouplingNavierSlip: Averaging Strategy not set!");
  else
    FOUR_C_THROW(
        "XFEM::LevelSetCouplingNavierSlip: You want to initialize another strategy than "
        "Xfluid_Sided?");
  return;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingNavierSlip::update_configuration_map_gp(
    double& kappa_m,         //< fluid sided weighting
    double& visc_m,          //< master sided dynamic viscosity
    double& visc_s,          //< slave sided dynamic viscosity
    double& density_m,       //< master sided density
    double& visc_stab_tang,  //< viscous tangential NIT Penalty scaling
    double& full_stab, const CORE::LINALG::Matrix<3, 1>& x, const CORE::Conditions::Condition* cond,
    DRT::Element* ele,   //< Element
    DRT::Element* bele,  //< Boundary Element
    double* funct,       //< local shape function for Gauss Point (from fluid element)
    double* derxy,  //< local derivatives of shape function for Gauss Point (from fluid element)
    CORE::LINALG::Matrix<3, 1>& rst_slave,  //< local coord of gp on slave boundary element
    CORE::LINALG::Matrix<3, 1>& normal,     //< normal at gp
    CORE::LINALG::Matrix<3, 1>& vel_m,      //< master velocity at gp
    double* fulltraction                    //< precomputed fsi traction (sigmaF n + gamma relvel)
)
{
  double dynvisc = (kappa_m * visc_m + (1.0 - kappa_m) * visc_s);
  double sliplength = 0.0;
  GetSlipCoefficient(sliplength, x, cond);

  if (sliplength < 0.0) FOUR_C_THROW("The slip length can not be negative.");

  if (sliplength != 0.0)
  {
    double stabnit = 0.0;
    double stabadj = 0.0;
    XFEM::UTILS::GetNavierSlipStabilizationParameters(
        visc_stab_tang, dynvisc, sliplength, stabnit, stabadj);
    configuration_map_[INPAR::XFEM::F_Pen_t_Row].second = stabnit;
    configuration_map_[INPAR::XFEM::F_Con_t_Row] =
        std::pair<bool, double>(true, -stabnit);  //+sign for penalty!
    configuration_map_[INPAR::XFEM::F_Con_t_Col] =
        std::pair<bool, double>(true, sliplength / dynvisc);
    configuration_map_[INPAR::XFEM::F_Adj_t_Row].second = stabadj;
    configuration_map_[INPAR::XFEM::FStr_Adj_t_Col] = std::pair<bool, double>(true, sliplength);
  }
  else
  {
    configuration_map_[INPAR::XFEM::F_Pen_t_Row].second = visc_stab_tang;
    configuration_map_[INPAR::XFEM::F_Con_t_Row] = std::pair<bool, double>(false, 0.0);
    configuration_map_[INPAR::XFEM::F_Con_t_Col] = std::pair<bool, double>(false, 0.0);
    configuration_map_[INPAR::XFEM::F_Adj_t_Row].second = 1.0;
    configuration_map_[INPAR::XFEM::FStr_Adj_t_Col] = std::pair<bool, double>(false, 0.0);
  }

  // Configuration of Penalty Terms
  configuration_map_[INPAR::XFEM::F_Pen_n_Row].second =
      visc_stab_tang;  // full_stab <-- to keep results!

  return;
}

FOUR_C_NAMESPACE_CLOSE