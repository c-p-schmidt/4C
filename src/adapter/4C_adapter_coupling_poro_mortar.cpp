/*---------------------------------------------------------------------*/
/*! \file

\brief coupling adapter for porous meshtying

Masterthesis of h.Willmann under supervision of Anh-Tu Vuong
and Johannes Kremheller, Originates from ADAPTER::CouplingNonLinMortar

\level 3


*/
/*---------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  includes                                                  ager 10/15|
 *----------------------------------------------------------------------*/
// lib
#include "4C_adapter_coupling_poro_mortar.hpp"

#include "4C_contact_element.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_node.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_nurbs_discret.hpp"
#include "4C_nurbs_discret_control_point.hpp"
#include "4C_nurbs_discret_knotvector.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor                                                      ager 10/15|
 *----------------------------------------------------------------------*/
ADAPTER::CouplingPoroMortar::CouplingPoroMortar(int spatial_dimension,
    Teuchos::ParameterList mortar_coupling_params, Teuchos::ParameterList contact_dynamic_params,
    CORE::FE::ShapeFunctionType shape_function_type)
    : CouplingNonLinMortar(
          spatial_dimension, mortar_coupling_params, contact_dynamic_params, shape_function_type),
      firstinit_(false),
      slavetype_(-1),
      mastertype_(-1)
{
  // empty...
}

/*----------------------------------------------------------------------*
 |  Read Mortar Condition                                     ager 10/15|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingPoroMortar::ReadMortarCondition(Teuchos::RCP<DRT::Discretization> masterdis,
    Teuchos::RCP<DRT::Discretization> slavedis, std::vector<int> coupleddof,
    const std::string& couplingcond, Teuchos::ParameterList& input,
    std::map<int, DRT::Node*>& mastergnodes, std::map<int, DRT::Node*>& slavegnodes,
    std::map<int, Teuchos::RCP<DRT::Element>>& masterelements,
    std::map<int, Teuchos::RCP<DRT::Element>>& slaveelements)
{
  // Call Base Class
  CouplingNonLinMortar::ReadMortarCondition(masterdis, slavedis, coupleddof, couplingcond, input,
      mastergnodes, slavegnodes, masterelements, slaveelements);

  // Set Problem Type to Poro
  switch (GLOBAL::Problem::Instance()->GetProblemType())
  {
    case GLOBAL::ProblemType::poroelast:
      input.set<int>("PROBTYPE", INPAR::CONTACT::poroelast);
      break;
    case GLOBAL::ProblemType::poroscatra:
      input.set<int>("PROBTYPE", INPAR::CONTACT::poroscatra);
      break;
    default:
      FOUR_C_THROW("Invalid poro problem is specified");
      break;
  }

  // porotimefac = 1/(theta*dt) --- required for derivation of structural displacements!
  const Teuchos::ParameterList& stru = GLOBAL::Problem::Instance()->structural_dynamic_params();
  double porotimefac =
      1 / (stru.sublist("ONESTEPTHETA").get<double>("THETA") * stru.get<double>("TIMESTEP"));
  input.set<double>("porotimefac", porotimefac);
  const Teuchos::ParameterList& porodyn = GLOBAL::Problem::Instance()->poroelast_dynamic_params();
  input.set<bool>("CONTACTNOPEN",
      CORE::UTILS::IntegralValue<int>(porodyn, "CONTACTNOPEN"));  // used in the integrator
  if (!CORE::UTILS::IntegralValue<int>(porodyn, "CONTACTNOPEN"))
    FOUR_C_THROW("Set CONTACTNOPEN for Poroelastic meshtying!");
}

/*----------------------------------------------------------------------*
 |  Add Mortar Elements                                        ager 10/15|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingPoroMortar::AddMortarElements(Teuchos::RCP<DRT::Discretization> masterdis,
    Teuchos::RCP<DRT::Discretization> slavedis, Teuchos::ParameterList& input,
    std::map<int, Teuchos::RCP<DRT::Element>>& masterelements,
    std::map<int, Teuchos::RCP<DRT::Element>>& slaveelements,
    Teuchos::RCP<CONTACT::Interface>& interface, int numcoupleddof)
{
  bool isnurbs = input.get<bool>("NURBS");

  // get problem dimension (2D or 3D) and create (MORTAR::Interface)
  const int dim = GLOBAL::Problem::Instance()->NDim();

  // We need to determine an element offset to start the numbering of the slave
  // mortar elements AFTER the master mortar elements in order to ensure unique
  // eleIDs in the interface discretization. The element offset equals the
  // overall number of master mortar elements (which is not equal to the number
  // of elements in the field that is chosen as master side).
  //
  // If masterdis==slavedis, the element numbering is right without offset
  int eleoffset = 0;
  if (masterdis.get() != slavedis.get())
  {
    int nummastermtreles = masterelements.size();
    comm_->SumAll(&nummastermtreles, &eleoffset, 1);
  }

  // feeding master elements to the interface
  std::map<int, Teuchos::RCP<DRT::Element>>::const_iterator elemiter;
  for (elemiter = masterelements.begin(); elemiter != masterelements.end(); ++elemiter)
  {
    Teuchos::RCP<DRT::Element> ele = elemiter->second;
    Teuchos::RCP<CONTACT::Element> cele = Teuchos::rcp(new CONTACT::Element(
        ele->Id(), ele->Owner(), ele->Shape(), ele->num_node(), ele->NodeIds(), false, isnurbs));

    Teuchos::RCP<DRT::FaceElement> faceele = Teuchos::rcp_dynamic_cast<DRT::FaceElement>(ele, true);
    if (faceele == Teuchos::null) FOUR_C_THROW("Cast to FaceElement failed!");
    cele->PhysType() = MORTAR::Element::other;

    std::vector<Teuchos::RCP<CORE::Conditions::Condition>> porocondvec;
    masterdis->GetCondition("PoroCoupling", porocondvec);
    for (unsigned int i = 0; i < porocondvec.size(); ++i)
    {
      std::map<int, Teuchos::RCP<DRT::Element>>::const_iterator eleitergeometry;
      for (eleitergeometry = porocondvec[i]->Geometry().begin();
           eleitergeometry != porocondvec[i]->Geometry().end(); ++eleitergeometry)
      {
        if (faceele->parent_element()->Id() == eleitergeometry->second->Id())
        {
          if (mastertype_ == 0)
            FOUR_C_THROW(
                "struct and poro master elements on the same processor - no mixed interface "
                "supported");
          cele->PhysType() = MORTAR::Element::poro;
          mastertype_ = 1;
          break;
        }
      }
    }
    if (cele->PhysType() == MORTAR::Element::other)
    {
      if (mastertype_ == 1)
        FOUR_C_THROW(
            "struct and poro master elements on the same processor - no mixed interface supported");
      cele->PhysType() = MORTAR::Element::structure;
      mastertype_ = 0;
    }

    cele->set_parent_master_element(faceele->parent_element(), faceele->FaceParentNumber());

    if (isnurbs)
    {
      Teuchos::RCP<DRT::NURBS::NurbsDiscretization> nurbsdis =
          Teuchos::rcp_dynamic_cast<DRT::NURBS::NurbsDiscretization>(masterdis);

      Teuchos::RCP<DRT::NURBS::Knotvector> knots = (*nurbsdis).GetKnotVector();
      std::vector<CORE::LINALG::SerialDenseVector> parentknots(dim);
      std::vector<CORE::LINALG::SerialDenseVector> mortarknots(dim - 1);

      Teuchos::RCP<DRT::FaceElement> faceele =
          Teuchos::rcp_dynamic_cast<DRT::FaceElement>(ele, true);
      double normalfac = 0.0;
      bool zero_size = knots->get_boundary_ele_and_parent_knots(parentknots, mortarknots, normalfac,
          faceele->ParentMasterElement()->Id(), faceele->FaceMasterNumber());

      // store nurbs specific data to node
      cele->ZeroSized() = zero_size;
      cele->Knots() = mortarknots;
      cele->NormalFac() = normalfac;
    }

    interface->add_element(cele);
  }

  // feeding slave elements to the interface
  for (elemiter = slaveelements.begin(); elemiter != slaveelements.end(); ++elemiter)
  {
    Teuchos::RCP<DRT::Element> ele = elemiter->second;

    Teuchos::RCP<CONTACT::Element> cele = Teuchos::rcp(new CONTACT::Element(
        ele->Id(), ele->Owner(), ele->Shape(), ele->num_node(), ele->NodeIds(), true, isnurbs));

    Teuchos::RCP<DRT::FaceElement> faceele = Teuchos::rcp_dynamic_cast<DRT::FaceElement>(ele, true);
    if (faceele == Teuchos::null) FOUR_C_THROW("Cast to FaceElement failed!");
    cele->PhysType() = MORTAR::Element::other;

    std::vector<Teuchos::RCP<CORE::Conditions::Condition>> porocondvec;
    masterdis->GetCondition("PoroCoupling", porocondvec);

    for (unsigned int i = 0; i < porocondvec.size(); ++i)
    {
      std::map<int, Teuchos::RCP<DRT::Element>>::const_iterator eleitergeometry;
      for (eleitergeometry = porocondvec[i]->Geometry().begin();
           eleitergeometry != porocondvec[i]->Geometry().end(); ++eleitergeometry)
      {
        if (faceele->parent_element()->Id() == eleitergeometry->second->Id())
        {
          if (slavetype_ == 0)
            FOUR_C_THROW(
                "struct and poro slave elements on the same processor - no mixed interface "
                "supported");
          cele->PhysType() = MORTAR::Element::poro;
          slavetype_ = 1;
          break;
        }
      }
    }
    if (cele->PhysType() == MORTAR::Element::other)
    {
      if (slavetype_ == 1)
        FOUR_C_THROW(
            "struct and poro slave elements on the same processor - no mixed interface supported");
      cele->PhysType() = MORTAR::Element::structure;
      slavetype_ = 0;
    }
    cele->set_parent_master_element(faceele->parent_element(), faceele->FaceParentNumber());

    if (isnurbs)
    {
      Teuchos::RCP<DRT::NURBS::NurbsDiscretization> nurbsdis =
          Teuchos::rcp_dynamic_cast<DRT::NURBS::NurbsDiscretization>(slavedis);

      Teuchos::RCP<DRT::NURBS::Knotvector> knots = (*nurbsdis).GetKnotVector();
      std::vector<CORE::LINALG::SerialDenseVector> parentknots(dim);
      std::vector<CORE::LINALG::SerialDenseVector> mortarknots(dim - 1);

      Teuchos::RCP<DRT::FaceElement> faceele =
          Teuchos::rcp_dynamic_cast<DRT::FaceElement>(ele, true);
      double normalfac = 0.0;
      bool zero_size = knots->get_boundary_ele_and_parent_knots(parentknots, mortarknots, normalfac,
          faceele->ParentMasterElement()->Id(), faceele->FaceMasterNumber());

      // store nurbs specific data to node
      cele->ZeroSized() = zero_size;
      cele->Knots() = mortarknots;
      cele->NormalFac() = normalfac;
    }

    interface->add_element(cele);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  read mortar condition                                    Ager 02/16 |
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingPoroMortar::CreateStrategy(Teuchos::RCP<DRT::Discretization> masterdis,
    Teuchos::RCP<DRT::Discretization> slavedis, Teuchos::ParameterList& input, int numcoupleddof)
{
  // poro lagrange strategy:

  // get problem dimension (2D or 3D) and create (MORTAR::Interface)
  const int dim = GLOBAL::Problem::Instance()->NDim();

  // bools to decide which side is structural and which side is poroelastic to manage all 4
  // constellations
  // s-s, p-s, s-p, p-p
  bool poromaster = false;
  bool poroslave = false;
  bool structmaster = false;
  bool structslave = false;

  // wait for all processors to determine if they have poro or structural master or slave elements
  comm_->Barrier();
  std::vector<int> slaveTypeList(comm_->NumProc());
  std::vector<int> masterTypeList(comm_->NumProc());
  comm_->GatherAll(&slavetype_, slaveTypeList.data(), 1);
  comm_->GatherAll(&mastertype_, masterTypeList.data(), 1);
  comm_->Barrier();

  for (int i = 0; i < comm_->NumProc(); ++i)
  {
    switch (slaveTypeList[i])
    {
      case -1:
        break;
      case 1:
        if (structslave)
          FOUR_C_THROW(
              "struct and poro slave elements on the same adapter - no mixed interface supported");
        // adjust FOUR_C_THROW text, when more than one interface is supported
        poroslave = true;
        break;
      case 0:
        if (poroslave)
          FOUR_C_THROW(
              "struct and poro slave elements on the same adapter - no mixed interface supported");
        structslave = true;
        break;
      default:
        FOUR_C_THROW("this cannot happen");
        break;
    }
  }

  for (int i = 0; i < comm_->NumProc(); ++i)
  {
    switch (masterTypeList[i])
    {
      case -1:
        break;
      case 1:
        if (structmaster)
          FOUR_C_THROW(
              "struct and poro master elements on the same adapter - no mixed interface supported");
        // adjust FOUR_C_THROW text, when more than one interface is supported
        poromaster = true;
        break;
      case 0:
        if (poromaster)
          FOUR_C_THROW(
              "struct and poro master elements on the same adapter - no mixed interface supported");
        structmaster = true;
        break;
      default:
        FOUR_C_THROW("this cannot happen");
        break;
    }
  }

  const Teuchos::ParameterList& stru = GLOBAL::Problem::Instance()->structural_dynamic_params();
  double theta = stru.sublist("ONESTEPTHETA").get<double>("THETA");
  // what if problem is static ? there should be an error for previous line called in a dyna_statics
  // problem and not a value of 0.5 a proper disctinction is necessary if poro meshtying is expanded
  // to other time integration strategies

  if (CORE::UTILS::IntegralValue<INPAR::STR::DynamicType>(stru, "DYNAMICTYP") ==
      INPAR::STR::dyna_statics)
  {
    theta = 1.0;
  }
  std::vector<Teuchos::RCP<CONTACT::Interface>> interfaces;
  interfaces.push_back(interface_);
  double alphaf = 1.0 - theta;

  // build the correct data container
  Teuchos::RCP<CONTACT::AbstractStratDataContainer> data_ptr =
      Teuchos::rcp(new CONTACT::AbstractStratDataContainer());
  // create contact poro lagrange strategy for mesh tying
  porolagstrategy_ = Teuchos::rcp(
      new CONTACT::LagrangeStrategyPoro(data_ptr, masterdis->dof_row_map(), masterdis->NodeRowMap(),
          input, interfaces, dim, comm_, alphaf, numcoupleddof, poroslave, poromaster));

  porolagstrategy_->Setup(false, true);
  porolagstrategy_->PoroMtInitialize();

  firstinit_ = true;

  return;
}


/*----------------------------------------------------------------------*
 |  complete interface (also print and parallel redist.)      Ager 02/16|
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingPoroMortar::CompleteInterface(
    Teuchos::RCP<DRT::Discretization> masterdis, Teuchos::RCP<CONTACT::Interface>& interface)
{
  // finalize the contact interface construction
  int maxdof = masterdis->dof_row_map()->MaxAllGID();
  interface->fill_complete(true, maxdof);

  // interface->create_volume_ghosting(*masterdis);

  // store old row maps (before parallel redistribution)
  slavedofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface->SlaveRowDofs()));
  masterdofrowmap_ = Teuchos::rcp(new Epetra_Map(*interface->MasterRowDofs()));
  slavenoderowmap_ = Teuchos::rcp(new Epetra_Map(*interface->SlaveRowNodes()));

  // print parallel distribution
  interface->print_parallel_distribution();

  // TODO: is this possible?
  //**********************************************************************
  // PARALLEL REDISTRIBUTION OF INTERFACE
  //**********************************************************************
  //  if (parredist && comm_->NumProc()>1)
  //  {
  //    // redistribute optimally among all procs
  //    interface->Redistribute(1);
  //
  //    // call fill complete again
  //    interface->fill_complete();
  //
  //    // print parallel distribution again
  //    interface->print_parallel_distribution();
  //  }

  // store interface
  interface_ = interface;

  // create binary search tree
  interface_->CreateSearchTree();

  return;
}


/*----------------------------------------------------------------------*
 |  evaluate blockmatrices for poro meshtying             ager 10/15    |
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingPoroMortar::EvaluatePoroMt(Teuchos::RCP<Epetra_Vector> fvel,
    Teuchos::RCP<Epetra_Vector> svel, Teuchos::RCP<Epetra_Vector> fpres,
    Teuchos::RCP<Epetra_Vector> sdisp, const Teuchos::RCP<DRT::Discretization> sdis,
    Teuchos::RCP<CORE::LINALG::SparseMatrix>& f, Teuchos::RCP<CORE::LINALG::SparseMatrix>& k_fs,
    Teuchos::RCP<Epetra_Vector>& frhs, CORE::ADAPTER::Coupling& coupfs,
    Teuchos::RCP<const Epetra_Map> fdofrowmap)
{
  // safety check
  check_setup();

  // write interface values into poro contact interface data containers for integration in contact
  // integrator
  porolagstrategy_->set_state(MORTAR::state_fvelocity, *fvel);
  porolagstrategy_->set_state(MORTAR::state_svelocity, *svel);
  porolagstrategy_->set_state(MORTAR::state_fpressure, *fpres);
  porolagstrategy_->set_state(MORTAR::state_new_displacement, *sdisp);

  // store displacements of parent elements for deformation gradient determinant and its
  // linearization
  porolagstrategy_->SetParentState("displacement", sdisp, sdis);

  interface_->Initialize();
  // in the end of Evaluate coupling condition residuals and linearizations are computed in contact
  // integrator
  interface_->Evaluate();

  porolagstrategy_->poro_mt_prepare_fluid_coupling();
  porolagstrategy_->PoroInitialize(coupfs, fdofrowmap, firstinit_);
  if (firstinit_) firstinit_ = false;

  // do system matrix manipulations
  porolagstrategy_->evaluate_poro_no_pen_contact(k_fs, f, frhs);
  return;
}  // ADAPTER::CouplingNonLinMortar::EvaluatePoroMt()


/*----------------------------------------------------------------------*
 |  update poro meshtying quantities                      ager 10/15    |
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingPoroMortar::UpdatePoroMt()
{
  // safety check
  check_setup();

  porolagstrategy_->PoroMtUpdate();
  return;
}  // ADAPTER::CouplingNonLinMortar::UpdatePoroMt()

/*----------------------------------------------------------------------*
 |  recover fluid coupling lagrange multiplier            ager 10/15    |
 *----------------------------------------------------------------------*/
void ADAPTER::CouplingPoroMortar::recover_fluid_lm_poro_mt(
    Teuchos::RCP<Epetra_Vector> disi, Teuchos::RCP<Epetra_Vector> veli)
{
  // safety check
  check_setup();

  porolagstrategy_->RecoverPoroNoPen(disi, veli);
  return;
}  // ADAPTER::CouplingNonLinMortar::recover_fluid_lm_poro_mt

FOUR_C_NAMESPACE_CLOSE