/*---------------------------------------------------------------------*/
/*! \file

\brief a class to manage one discretization with changing dofs in xfem context

\level 1


*/
/*----------------------------------------------------------------------*/

#include "4C_lib_discret_xfem.hpp"

#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_xfem_dofset.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                             ager 11/14|
 |  comm             (in)  a communicator object                        |
 *----------------------------------------------------------------------*/
DRT::DiscretizationXFEM::DiscretizationXFEM(const std::string name, Teuchos::RCP<Epetra_Comm> comm)
    : DiscretizationFaces(name, comm),
      initialized_(false),
      initialfulldofrowmap_(Teuchos::null),
      initialpermdofrowmap_(Teuchos::null)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Finalize construction (public)                             ager 11/14|
 *----------------------------------------------------------------------*/
int DRT::DiscretizationXFEM::InitialFillComplete(const std::vector<int>& nds,
    bool assigndegreesoffreedom, bool initelements, bool doboundaryconditions)
{
  // Call from BaseClass
  int val = DRT::Discretization::fill_complete(
      assigndegreesoffreedom, initelements, doboundaryconditions);

  if (!assigndegreesoffreedom)
    FOUR_C_THROW(
        "DiscretizationXFEM: Call InitialFillComplete() with assigndegreesoffreedom = true!");

  // Store initial dofs of the discretisation
  store_initial_dofs(nds);
  return val;
}

/*----------------------------------------------------------------------*
 |  checks if Discretization is initialized (protected)  ager 11/14|
 *----------------------------------------------------------------------*/
bool DRT::DiscretizationXFEM::Initialized() const
{
  if (!initialized_)
    FOUR_C_THROW("DiscretizationXFEM is not initialized! - Call InitialFillComplete() once!");
  return initialized_;
}

/*----------------------------------------------------------------------*
 |  Store Initial Dofs (private)                               ager 11/14|
 *----------------------------------------------------------------------*/
void DRT::DiscretizationXFEM::store_initial_dofs(const std::vector<int>& nds)
{
  if (nds.size() != 1)
    FOUR_C_THROW(
        "DiscretizationXFEM: At the moment just one initial dofset to be initialized is supported "
        "by DiscretisationXFEM!");

  // store copy of initial dofset
  initialdofsets_.clear();

  initialdofsets_.push_back(
      Teuchos::rcp_dynamic_cast<CORE::Dofsets::DofSet>(dofsets_[nds[0]], true)->Clone());

  // store map required for export to active dofs
  if (initialdofsets_.size() > 1)
    FOUR_C_THROW(
        "DiscretizationXFEM: At the moment just one initial dofset is supported by "
        "DiscretisationXFEM!");

  Teuchos::RCP<CORE::Dofsets::FixedSizeDofSet> fsds =
      Teuchos::rcp_dynamic_cast<CORE::Dofsets::FixedSizeDofSet>(initialdofsets_[0]);
  if (fsds == Teuchos::null)
    FOUR_C_THROW("DiscretizationXFEM: Cast to CORE::Dofsets::FixedSizeDofSet failed!");

  Teuchos::RCP<XFEM::XFEMDofSet> xfds =
      Teuchos::rcp_dynamic_cast<XFEM::XFEMDofSet>(initialdofsets_[0]);
  if (xfds != Teuchos::null)
    FOUR_C_THROW("DiscretizationXFEM: Initial Dofset shouldn't be a XFEM::XFEMDofSet!");

  int numdofspernode = 0;
  fsds->get_reserved_max_num_dofper_node(numdofspernode);

  if (NumMyColNodes() == 0) FOUR_C_THROW("no column node on this proc available!");
  int numdofspernodedofset = fsds->NumDof(lColNode(0));
  int numdofsetspernode = 0;

  if (numdofspernode % numdofspernodedofset)
    FOUR_C_THROW("DiscretizationXFEM: Dividing numdofspernode / numdofspernodedofset failed!");
  else
    numdofsetspernode = numdofspernode / numdofspernodedofset;

  initialfulldofrowmap_ =
      extend_map(fsds->dof_row_map(), numdofspernodedofset, numdofsetspernode, true);
  initialpermdofrowmap_ =
      extend_map(fsds->dof_row_map(), numdofspernodedofset, numdofsetspernode, false);

  initialized_ = true;

  return;
}

/*------------------------------------------------------------------------------*
 * Export Vector with initialdofrowmap (all nodes have one dofset) - to Vector  |
 * with all active dofs (public)                                       ager 11/14|
 *  *---------------------------------------------------------------------------*/
void DRT::DiscretizationXFEM::export_initialto_active_vector(
    Teuchos::RCP<const Epetra_Vector>& initialvec, Teuchos::RCP<Epetra_Vector>& activevec)
{
  // Is the discretization initialized?
  Initialized();

  Teuchos::RCP<Epetra_Vector> fullvec =
      Teuchos::rcp(new Epetra_Vector(*initialpermdofrowmap_, true));

  {  // Export manually as target.Map().UniqueGIDs() gives = true, although this shouldn't be the
     // case
    //(UniqueGIDs() just checks if gid occurs on more procs!)
    if (initialvec->Comm().NumProc() == 1 &&
        activevec->Comm().NumProc() == 1)  // for one proc , Export works fine!
    {
      CORE::LINALG::Export(*initialvec, *fullvec);
    }
    else
    {
      Epetra_Import importer(fullvec->Map(), initialvec->Map());
      int err = fullvec->Import(*initialvec, importer, Insert);
      if (err) FOUR_C_THROW("Export using exporter returned err=%d", err);
    }
  }
  fullvec->ReplaceMap(*initialfulldofrowmap_);  /// replace |1 2 3 4|1 2 3 4| -> |1 2 3 4|5 6 7 8|
  CORE::LINALG::Export(*fullvec, *activevec);
}

/*------------------------------------------------------------------------------*
 * Export Vector with initialdofrowmap (all nodes have one dofset) - to Vector  |
 * with all active dofs (public)                                       ager 11/14|
 *  *---------------------------------------------------------------------------*/
void DRT::DiscretizationXFEM::export_activeto_initial_vector(
    Teuchos::RCP<const Epetra_Vector> activevec, Teuchos::RCP<Epetra_Vector> initialvec)
{
  // Is the discretization initialized?
  Initialized();

  CORE::LINALG::Export(*activevec, *initialvec);
}

/*----------------------------------------------------------------------*
 |  get dof row map (public)                                 ager 11/14 |
 *----------------------------------------------------------------------*/
const Epetra_Map* DRT::DiscretizationXFEM::InitialDofRowMap(unsigned nds) const
{
  Initialized();
  FOUR_C_ASSERT(nds < initialdofsets_.size(), "undefined initial dof set");

  return initialdofsets_[nds]->dof_row_map();
}


/*----------------------------------------------------------------------*
 |  get dof column map (public)                              ager 11/14 |
 *----------------------------------------------------------------------*/
const Epetra_Map* DRT::DiscretizationXFEM::InitialDofColMap(unsigned nds) const
{
  Initialized();
  FOUR_C_ASSERT(nds < initialdofsets_.size(), "undefined initial dof set");

  return initialdofsets_[nds]->DofColMap();
}

/*---------------------------------------------------------------------------*
 * Takes dof_row_map with just one xfem-Dofset and duplicates                  |
 * the dof gids for export to active dofs                          ager 11/14|
 *---------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> DRT::DiscretizationXFEM::extend_map(
    const Epetra_Map* srcmap, int numdofspernodedofset, int numdofsets, bool uniquenumbering)
{
  int numsrcelements = srcmap->NumMyElements();
  const int* srcgids = srcmap->MyGlobalElements();
  std::vector<int> dstgids;
  for (int i = 0; i < numsrcelements; i += numdofspernodedofset)
  {
    if (numsrcelements < i + numdofspernodedofset) FOUR_C_THROW("extend_map(): Check your srcmap!");
    for (int dofset = 0; dofset < numdofsets; ++dofset)
    {
      for (int dof = 0; dof < numdofspernodedofset; ++dof)
      {
        dstgids.push_back(srcgids[i + dof] + uniquenumbering * dofset * numdofspernodedofset);
      }
    }
  }

  return Teuchos::rcp(new Epetra_Map(-1, dstgids.size(), dstgids.data(), 0, srcmap->Comm()));
}

/*----------------------------------------------------------------------*
 |  set a reference to a data vector (public)                mwgee 12/06|
 *----------------------------------------------------------------------*/
void DRT::DiscretizationXFEM::SetInitialState(
    unsigned nds, const std::string& name, Teuchos::RCP<const Epetra_Vector> state)
{
  TEUCHOS_FUNC_TIME_MONITOR("DRT::DiscretizationXFEM::SetInitialState");

  if (!HaveDofs()) FOUR_C_THROW("fill_complete() was not called");
  const Epetra_Map* colmap = InitialDofColMap(nds);
  const Epetra_BlockMap& vecmap = state->Map();

  if (state_.size() <= nds) state_.resize(nds + 1);

  // if it's already in column map just set a reference
  // This is a rough test, but it might be ok at this place. It is an
  // error anyway to hand in a vector that is not related to our dof
  // maps.
  if (vecmap.PointSameAs(*colmap))
  {
    state_[nds][name] = state;
  }
  else  // if it's not in column map export and allocate
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (not InitialDofRowMap(nds)->SameAs(state->Map()))
    {
      FOUR_C_THROW(
          "row map of discretization and state vector %s are different. This is a fatal bug!",
          name.c_str());
    }
#endif
    Teuchos::RCP<Epetra_Vector> tmp = CORE::LINALG::CreateVector(*colmap, false);
    CORE::LINALG::Export(*state, *tmp);
    state_[nds][name] = tmp;
  }
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool DRT::DiscretizationXFEM::IsEqualXDofSet(int nds, const XFEM::XFEMDofSet& xdofset_new) const
{
  const XFEM::XFEMDofSet* xdofset_old = dynamic_cast<XFEM::XFEMDofSet*>(dofsets_[nds].get());
  if (not xdofset_old) return false;

  return ((*xdofset_old) == xdofset_new);
}

FOUR_C_NAMESPACE_CLOSE