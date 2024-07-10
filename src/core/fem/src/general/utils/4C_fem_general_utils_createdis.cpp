/*----------------------------------------------------------------------*/
/*! \file

\brief utility functions for automatic creation of a discretization
       from an existing one (e.g. ALE from Fluid)

\level 1


*/
/*----------------------------------------------------------------------*/

#include "4C_fem_general_utils_createdis.hpp"

#include "4C_fem_dofset_transparent_independent.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_rebalance_binning_based.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::FE::DiscretizationCreatorBase::initial_checks(
    const Core::FE::Discretization& sourcedis, const Core::FE::Discretization& targetdis) const
{
  // are the source and target discretizations ready?
  if (!sourcedis.filled()) FOUR_C_THROW("The source discretization is not filled!");
  if (!targetdis.filled()) FOUR_C_THROW("The target discretization is not filled!");

  // is the target discretization really empty?
  if (targetdis.num_global_elements() or targetdis.num_global_nodes())
  {
    FOUR_C_THROW("There are %d elements and %d nodes in target discretization. Panic.",
        targetdis.num_global_elements(), targetdis.num_global_nodes());
  }
  // Ok. Let's go on
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::FE::DiscretizationCreatorBase::create_nodes(const Core::FE::Discretization& sourcedis,
    Core::FE::Discretization& targetdis, const std::set<int>& rownodeset,
    const std::set<int>& colnodeset, const bool isnurbsdis, const bool buildimmersednode) const
{
  // prepare some variables we need
  int myrank = targetdis.get_comm().MyPID();
  const Epetra_Map* sourcenoderowmap = sourcedis.node_row_map();

  // construct nodes / control points in the new discretization
  if (isnurbsdis == false)
  {
    for (int i = 0; i < sourcenoderowmap->NumMyElements(); ++i)
    {
      int gid = sourcenoderowmap->GID(i);
      if (rownodeset.find(gid) != rownodeset.end())
      {
        Core::Nodes::Node* node_to_create = sourcedis.l_row_node(i);
        if (!buildimmersednode)
          targetdis.add_node(Teuchos::rcp(new Core::Nodes::Node(gid, node_to_create->x(), myrank)));
        else
          targetdis.add_node(
              Teuchos::rcp(new Core::Nodes::ImmersedNode(gid, node_to_create->x(), myrank)));
      }
    }
  }
  else
  {
    for (int i = 0; i < sourcenoderowmap->NumMyElements(); ++i)
    {
      const int gid = sourcenoderowmap->GID(i);
      if (rownodeset.find(gid) != rownodeset.end())
      {
        Core::FE::Nurbs::ControlPoint* node_to_create =
            dynamic_cast<Core::FE::Nurbs::ControlPoint*>(sourcedis.l_row_node(i));
        targetdis.add_node(Teuchos::rcp(new Core::FE::Nurbs::ControlPoint(
            gid, node_to_create->x(), node_to_create->w(), myrank)));
      }
    }
  }

  // ensure reset() is called on targetdis on all procs (including procs without rownodes)
  targetdis.check_filled_globally();

  return;
}  // Core::FE::DiscretizationCreatorBase::CreateNodes

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> Core::FE::DiscretizationCreatorBase::create_map(
    std::set<int>& gidset, const Core::FE::Discretization& targetdis) const
{
  // we get the node maps almost for free
  std::vector<int> targetgidvec(gidset.begin(), gidset.end());
  gidset.clear();

  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
      new Epetra_Map(-1, targetgidvec.size(), targetgidvec.data(), 0, targetdis.get_comm()));
  targetgidvec.clear();

  return map;
}  // Core::FE::DiscretizationCreatorBase::CreateMap

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::FE::DiscretizationCreatorBase::copy_conditions(const Core::FE::Discretization& sourcedis,
    Core::FE::Discretization& targetdis,
    const std::map<std::string, std::string>& conditions_to_copy) const
{
  // copy selected conditions to the new discretization (and rename them if desired)
  for (const auto& condition_pair : conditions_to_copy)
  {
    std::vector<Core::Conditions::Condition*> conds;
    sourcedis.get_condition(condition_pair.first, conds);
    for (const auto& cond : conds)
    {
      // We use the same nodal ids and therefore we can just copy the conditions.
      // The string-map gives the new condition names
      // (e.g. renaming from TransportDirichlet to Dirichlet)
      targetdis.set_condition(condition_pair.second, cond->copy_without_geometry());
    }
    conds.clear();
  }
}  // Core::FE::DiscretizationCreatorBase::CopyConditions

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::FE::Discretization>
Core::FE::DiscretizationCreatorBase::create_matching_discretization(
    const Teuchos::RCP<Core::FE::Discretization>& sourcedis, const std::string& targetdisname,
    bool clonedofs, bool assigndegreesoffreedom, bool initelements, bool doboundaryconditions) const
{
  // initialize identical clone discretization
  Teuchos::RCP<Epetra_Comm> comm = Teuchos::rcp(sourcedis->get_comm().Clone());
  Teuchos::RCP<Core::FE::Discretization> targetdis =
      Teuchos::rcp(new Core::FE::Discretization(targetdisname, comm, sourcedis->n_dim()));

  // clone nodes
  for (int i = 0; i < sourcedis->node_col_map()->NumMyElements(); ++i)
  {
    Core::Nodes::Node* node = sourcedis->l_col_node(i);
    if (!node) FOUR_C_THROW("Cannot find node with lid %", i);
    Teuchos::RCP<Core::Nodes::Node> newnode = Teuchos::rcp(node->clone());
    targetdis->add_node(newnode);
  }

  // clone elements
  for (int i = 0; i < sourcedis->element_col_map()->NumMyElements(); ++i)
  {
    Core::Elements::Element* ele = sourcedis->l_col_element(i);
    if (!ele) FOUR_C_THROW("Cannot find element with lid %", i);
    Teuchos::RCP<Core::Elements::Element> newele = Teuchos::rcp(ele->clone());
    targetdis->add_element(newele);
  }

  // clone conditions (prescribed in input file)
  std::vector<std::string> allcond;
  sourcedis->get_condition_names(allcond);
  // loop all conditions types
  for (unsigned numcond = 0; numcond < allcond.size(); ++numcond)
  {
    // get condition
    std::vector<Core::Conditions::Condition*> actcond;
    sourcedis->get_condition(allcond[numcond], actcond);

    // loop all condition of the current type
    for (unsigned numactcond = 0; numactcond < actcond.size(); ++numactcond)
      targetdis->set_condition(allcond[numcond], actcond[numactcond]->copy_without_geometry());
  }

  // make auxiliary discretization have the same dofs as the coupling discretization
  if (clonedofs)
    targetdis->replace_dof_set(Teuchos::rcp(new Core::DOFSets::IndependentDofSet()), false);
  targetdis->fill_complete(assigndegreesoffreedom, initelements, doboundaryconditions);

  // at the end, we do several checks to ensure that we really have generated
  // an identical discretization
  if (not sourcedis->node_row_map()->SameAs(*(targetdis->node_row_map())))
    FOUR_C_THROW("NodeRowMaps of source and target discretization are different!");
  if (not sourcedis->node_col_map()->SameAs(*(targetdis->node_col_map())))
    FOUR_C_THROW("NodeColMaps of source and target discretization are different!");
  if (not sourcedis->element_row_map()->SameAs(*(targetdis->element_row_map())))
    FOUR_C_THROW("ElementRowMaps of source and target discretization are different!");
  if (not sourcedis->element_col_map()->SameAs(*(targetdis->element_col_map())))
    FOUR_C_THROW("ElementColMaps of source and target discretization are different!");
  if (clonedofs)
  {
    if (not sourcedis->dof_row_map()->SameAs(*(targetdis->dof_row_map())))
      FOUR_C_THROW("DofRowMaps of source and target discretization are different!");
    if (not sourcedis->dof_col_map()->SameAs(*(targetdis->dof_col_map())))
      FOUR_C_THROW("DofColMaps of source and target discretization are different!");
  }

  // return identical dis
  return targetdis;

}  // Core::FE::DiscretizationCreatorBase::create_matching_discretization

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::FE::DiscretizationCreatorBase::finalize(
    const Core::FE::Discretization& sourcedis, Core::FE::Discretization& targetdis) const
{
  // export according to previously filled maps
  targetdis.export_row_nodes(*targetnoderowmap_);
  targetdis.export_column_nodes(*targetnodecolmap_);
  targetdis.export_row_elements(*targetelerowmap_);
  targetdis.export_column_elements(*targetelecolmap_);
  targetdis.fill_complete(false, false, false);

  // extra work for NURBS discretizations

  // try to cast sourcedis to NurbsDiscretisation
  const Core::FE::Nurbs::NurbsDiscretization* nurbsdis_ptr =
      dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&sourcedis);

  if (nurbsdis_ptr != nullptr)
  {
    Core::FE::Nurbs::NurbsDiscretization* targetnurbsdis_ptr =
        dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&targetdis);

    if (targetnurbsdis_ptr == nullptr)
    {
      FOUR_C_THROW("Nurbs source discretization but no nurbs target discretization\n");
    }

    Teuchos::RCP<Core::FE::Nurbs::Knotvector> knots =
        Teuchos::rcp(new Core::FE::Nurbs::Knotvector(*(nurbsdis_ptr->get_knot_vector())));

    // reset offsets
    int smallest_gid_in_dis = targetnurbsdis_ptr->element_row_map()->MinAllGID();
    knots->finish_knots(smallest_gid_in_dis);

    targetnurbsdis_ptr->set_knot_vector(knots);
    targetnurbsdis_ptr->fill_complete();
  }

  // at the end, we do several checks to ensure that we really have identical
  // distributions of elements and nodes over processors (as expected!)
  // We do not perform this check if the new discretization is only a subset of the
  // source discretization.
  int sumeleskips = 0;
  int lnumeleskips = numeleskips_;
  sourcedis.get_comm().SumAll(&lnumeleskips, &sumeleskips, 1);

  if (sumeleskips == 0)
  {
    if (not sourcedis.node_row_map()->SameAs(*(targetdis.node_row_map())))
      FOUR_C_THROW("NodeRowMaps of source and target discretization are different!");
    if (not sourcedis.node_col_map()->SameAs(*(targetdis.node_col_map())))
      FOUR_C_THROW("NodeColMaps of source and target discretization are different!");
    if (not sourcedis.element_row_map()->SameAs(*(targetdis.element_row_map())))
      FOUR_C_THROW("ElementRowMaps of source and target discretization are different!");
    if (not sourcedis.element_col_map()->SameAs(*(targetdis.element_col_map())))
      FOUR_C_THROW("ElementColMaps of source and target discretization are different!");
  }
}  // Core::FE::DiscretizationCreatorBase::Finalize

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<Input::LineDefinition> Core::FE::valid_cloning_material_map_lines()
{
  return {Input::LineDefinition::Builder()
              .add_named_string("SRC_FIELD")
              .add_named_int("SRC_MAT")
              .add_named_string("TAR_FIELD")
              .add_named_int("TAR_MAT")
              .build()};
}

FOUR_C_NAMESPACE_CLOSE