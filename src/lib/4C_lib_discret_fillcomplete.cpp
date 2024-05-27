/*---------------------------------------------------------------------*/
/*! \file

\brief Setup of discretization including assignment of degrees of freedom

\level 0


*/
/*---------------------------------------------------------------------*/

#include "4C_comm_parobjectfactory.hpp"
#include "4C_io_pstream.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  Finalize construction (public)                           mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::Reset(bool killdofs, bool killcond)
{
  filled_ = false;
  if (killdofs)
  {
    havedof_ = false;
    for (unsigned i = 0; i < dofsets_.size(); ++i) dofsets_[i]->Reset();
  }

  elerowmap_ = Teuchos::null;
  elecolmap_ = Teuchos::null;
  elerowptr_.clear();
  elecolptr_.clear();
  noderowmap_ = Teuchos::null;
  nodecolmap_ = Teuchos::null;
  noderowptr_.clear();
  nodecolptr_.clear();

  // delete all old geometries that are attached to any conditions
  // as early as possible
  if (killcond)
  {
    std::multimap<std::string, Teuchos::RCP<CORE::Conditions::Condition>>::iterator fool;
    for (fool = condition_.begin(); fool != condition_.end(); ++fool)
    {
      fool->second->clear_geometry();
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Finalize construction (public)                           mwgee 11/06|
 *----------------------------------------------------------------------*/
int DRT::Discretization::fill_complete(
    bool assigndegreesoffreedom, bool initelements, bool doboundaryconditions)
{
  // my processor id
  const int myrank = Comm().MyPID();

  // print information to screen
  if (myrank == 0)
  {
    IO::cout(IO::verbose)
        << "\n+--------------------------------------------------------------------+" << IO::endl;
    IO::cout(IO::verbose) << "| fill_complete() on discretization " << std::setw(34) << std::left
                          << Name() << std::setw(1) << std::right << "|" << IO::endl;
  }

  // set all maps to Teuchos::null
  Reset(assigndegreesoffreedom, doboundaryconditions);

  // (re)build map of nodes noderowmap_, nodecolmap_, noderowptr and nodecolptr
  build_node_row_map();
  build_node_col_map();

  // (re)build map of elements elemap_
  build_element_row_map();
  build_element_col_map();

  // (re)construct element -> node pointers
  build_element_to_node_pointers();

  // (re)construct node -> element pointers
  build_node_to_element_pointers();

  // (re)construct element -> element pointers for interface-elements
  build_element_to_element_pointers();

  // set the flag indicating Filled()==true
  // as the following methods make use of maps
  // which we just built
  filled_ = true;

  // Assign degrees of freedom to elements and nodes
  if (assigndegreesoffreedom)
  {
    if (myrank == 0)
    {
      IO::cout(IO::verbose)
          << "| assign_degrees_of_freedom() ...                                       |"
          << IO::endl;
    }
    assign_degrees_of_freedom(0);
  }

  // call element routines to initialize
  if (initelements)
  {
    if (myrank == 0)
    {
      IO::cout(IO::verbose)
          << "| InitializeElements() ...                                           |" << IO::endl;
    }
    InitializeElements();
  }

  // (Re)build the geometry of the boundary conditions
  if (doboundaryconditions)
  {
    if (myrank == 0)
    {
      IO::cout(IO::verbose)
          << "| boundary_conditions_geometry() ...                                   |" << IO::endl;
    }

    boundary_conditions_geometry();
  }

  if (myrank == 0)
  {
    IO::cout(IO::verbose)
        << "+--------------------------------------------------------------------+" << IO::endl;
  }

  return 0;
}


/*----------------------------------------------------------------------*
 |  init elements (public)                                   mwgee 12/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::InitializeElements()
{
  if (!Filled()) FOUR_C_THROW("fill_complete was not called");

  CORE::COMM::ParObjectFactory::Instance().InitializeElements(*this);

  return;
}


/*----------------------------------------------------------------------*
 |  Build noderowmap_ (private)                              mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::build_node_row_map()
{
  const int myrank = Comm().MyPID();
  int nummynodes = 0;
  std::map<int, Teuchos::RCP<DRT::Node>>::iterator curr;
  for (curr = node_.begin(); curr != node_.end(); ++curr)
    if (curr->second->Owner() == myrank) ++nummynodes;
  std::vector<int> nodeids(nummynodes);
  noderowptr_.resize(nummynodes);

  int count = 0;
  for (curr = node_.begin(); curr != node_.end(); ++curr)
    if (curr->second->Owner() == myrank)
    {
      nodeids[count] = curr->second->Id();
      noderowptr_[count] = curr->second.get();
      ++count;
    }
  if (count != nummynodes) FOUR_C_THROW("Mismatch in no. of nodes");
  noderowmap_ = Teuchos::rcp(new Epetra_Map(-1, nummynodes, nodeids.data(), 0, Comm()));
  return;
}

/*----------------------------------------------------------------------*
 |  Build nodecolmap_ (private)                              mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::build_node_col_map()
{
  int nummynodes = (int)node_.size();
  std::vector<int> nodeids(nummynodes);
  nodecolptr_.resize(nummynodes);

  int count = 0;
  std::map<int, Teuchos::RCP<DRT::Node>>::iterator curr;
  for (curr = node_.begin(); curr != node_.end(); ++curr)
  {
    nodeids[count] = curr->second->Id();
    nodecolptr_[count] = curr->second.get();
    curr->second->SetLID(count);
    ++count;
  }
  if (count != nummynodes) FOUR_C_THROW("Mismatch in no. of nodes");
  nodecolmap_ = Teuchos::rcp(new Epetra_Map(-1, nummynodes, nodeids.data(), 0, Comm()));
  return;
}


/*----------------------------------------------------------------------*
 |  Build elerowmap_ (private)                                mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::build_element_row_map()
{
  const int myrank = Comm().MyPID();
  int nummyeles = 0;
  std::map<int, Teuchos::RCP<DRT::Element>>::iterator curr;
  for (curr = element_.begin(); curr != element_.end(); ++curr)
    if (curr->second->Owner() == myrank) nummyeles++;
  std::vector<int> eleids(nummyeles);
  elerowptr_.resize(nummyeles);
  int count = 0;
  for (curr = element_.begin(); curr != element_.end(); ++curr)
    if (curr->second->Owner() == myrank)
    {
      eleids[count] = curr->second->Id();
      elerowptr_[count] = curr->second.get();
      ++count;
    }
  if (count != nummyeles) FOUR_C_THROW("Mismatch in no. of elements");
  elerowmap_ = Teuchos::rcp(new Epetra_Map(-1, nummyeles, eleids.data(), 0, Comm()));
  return;
}

/*----------------------------------------------------------------------*
 |  Build elecolmap_ (private)                                mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::build_element_col_map()
{
  int nummyeles = (int)element_.size();
  std::vector<int> eleids(nummyeles);
  elecolptr_.resize(nummyeles);
  std::map<int, Teuchos::RCP<DRT::Element>>::iterator curr;
  int count = 0;
  for (curr = element_.begin(); curr != element_.end(); ++curr)
  {
    eleids[count] = curr->second->Id();
    elecolptr_[count] = curr->second.get();
    curr->second->SetLID(count);
    ++count;
  }
  if (count != nummyeles) FOUR_C_THROW("Mismatch in no. of elements");
  elecolmap_ = Teuchos::rcp(new Epetra_Map(-1, nummyeles, eleids.data(), 0, Comm()));
  return;
}

/*----------------------------------------------------------------------*
 |  Build ptrs element -> node (private)                      mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::build_element_to_node_pointers()
{
  std::map<int, Teuchos::RCP<DRT::Element>>::iterator elecurr;
  for (elecurr = element_.begin(); elecurr != element_.end(); ++elecurr)
  {
    bool success = elecurr->second->BuildNodalPointers(node_);
    if (!success) FOUR_C_THROW("Building element <-> node topology failed");
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Build ptrs element -> element (private)                      mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::build_element_to_element_pointers()
{
  std::map<int, Teuchos::RCP<DRT::Element>>::iterator elecurr;
  for (elecurr = element_.begin(); elecurr != element_.end(); ++elecurr)
  {
    bool success = elecurr->second->build_element_pointers(element_);
    if (!success) FOUR_C_THROW("Building element <-> element topology failed");
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Build ptrs node -> element (private)                      mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::build_node_to_element_pointers()
{
  std::map<int, Teuchos::RCP<DRT::Node>>::iterator nodecurr;
  for (nodecurr = node_.begin(); nodecurr != node_.end(); ++nodecurr)
    nodecurr->second->clear_my_element_topology();

  std::map<int, Teuchos::RCP<DRT::Element>>::iterator elecurr;
  for (elecurr = element_.begin(); elecurr != element_.end(); ++elecurr)
  {
    const int nnode = elecurr->second->num_node();
    const int* nodes = elecurr->second->NodeIds();
    for (int j = 0; j < nnode; ++j)
    {
      DRT::Node* node = gNode(nodes[j]);
      if (!node)
        FOUR_C_THROW("Node %d is not on this proc %d", j, Comm().MyPID());
      else
        node->AddElementPtr(elecurr->second.get());
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |  set degrees of freedom (protected)                       mwgee 03/07|
 *----------------------------------------------------------------------*/
int DRT::Discretization::assign_degrees_of_freedom(int start)
{
  if (!Filled()) FOUR_C_THROW("Filled()==false");
  if (!NodeRowMap()->UniqueGIDs()) FOUR_C_THROW("Nodal row map is not unique");
  if (!ElementRowMap()->UniqueGIDs()) FOUR_C_THROW("Element row map is not unique");

  // Set the havedof flag before dofs are assigned. Some dof set
  // implementations do query the discretization after the assignment has been
  // done and this query demands the havedof flag to be set. An unexpected
  // implicit dependency here.
  havedof_ = true;

  for (unsigned i = 0; i < dofsets_.size(); ++i)
    start = dofsets_[i]->assign_degrees_of_freedom(*this, i, start);
  return start;
}

FOUR_C_NAMESPACE_CLOSE