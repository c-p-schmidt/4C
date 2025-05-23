// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DOFSET_INTERFACE_HPP
#define FOUR_C_FEM_DOFSET_INTERFACE_HPP

#include "4C_config.hpp"

#include "4C_linalg_map.hpp"

#include <mpi.h>

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::Nodes
{
  class Node;
}

namespace Core::DOFSets
{

  /*! \brief Common interface class for all sets of degrees of freedom.
   *
   * This is a pure virtual class all classes managing sets of degrees of freedom
   * should inherit from.
   *

   * */
  class DofSetInterface
  {
   public:
    //! @name Construction

    /// Standard Constructor
    DofSetInterface() {};

    /// Destructor
    virtual ~DofSetInterface() = default;

    //@}


    //! @name Public Access Methods

    /// Get number of dofs for given node
    virtual int num_dof(
        const Core::Nodes::Node* node  ///< node, for which you want to know the number of dofs
    ) const = 0;

    /// Get number of dofs for given element
    virtual int num_dof(const Core::Elements::Element*
            element  ///< element, for which you want to know the number of dofs
    ) const = 0;

    /// get number of nodal dofs
    virtual int num_dof_per_node(
        const Core::Nodes::Node& node  ///< node, for which you want to know the number of dofs
    ) const = 0;

    /// Get the gid of a dof for given node
    virtual int dof(const Core::Nodes::Node* node,  ///< node, for which you want the dof positions
        int dof  ///< number of dof for which you want the dof position
    ) const = 0;

    /// Get the gid of a dof for given element
    virtual int dof(
        const Core::Elements::Element* element,  ///< element, for which you want the dof positions
        int dof                                  ///< number dof for which you want the dof position
    ) const = 0;

    /// Get the gid of all dofs of a node
    virtual std::vector<int> dof(
        const Core::Nodes::Node* node  ///< node, for which you want the dof positions
    ) const = 0;

    /// Get the gid of all dofs of a node
    virtual void dof(std::vector<int>& dof,  ///< vector of dof gids (to be filled)
        const Core::Nodes::Node* node,       ///< node, for which you want the dof positions
        unsigned nodaldofset  ///< number of nodal dof set of the node (currently !=0 only for XFEM)
    ) const = 0;

    /// Get the gid of all dofs of a element
    virtual std::vector<int> dof(const Core::Elements::Element* element) const = 0;

    /// Get the gid of all dofs of a node and the location matrix
    virtual void dof(const Core::Nodes::Node* node, std::vector<int>& lm) const = 0;

    /// Get the gid of all dofs of a node
    virtual void dof(const Core::Nodes::Node* node,  ///< node, for which you want the dof positions
        const unsigned startindex,  ///< first index of vector at which will be written to end
        std::vector<int>& lm        ///< already allocated vector to be filled with dof positions
    ) const = 0;

    /// Get the gid of all dofs of a element and the location matrix
    virtual void dof(const Core::Elements::Element* element, std::vector<int>& lm) const = 0;

    /// Get the GIDs of the first DOFs of a node of which the associated element is interested in
    virtual void dof(const Core::Elements::Element*
                         element,  ///< element which provides its expected number of DOFs per node
        const Core::Nodes::Node* node,  ///< node, for which you want the DOF positions
        std::vector<int>& lm  ///< already allocated vector to be filled with DOF positions
    ) const = 0;

    //@}


    //! @name Utility Methods

    /// Print this class
    virtual void print(std::ostream& os) const = 0;

    /// Print the dofsets in the static_dofsets_ list
    virtual void print_all_dofsets(MPI_Comm comm) const = 0;

    /// Returns true if filled
    virtual bool filled() const = 0;

    /// Add Dof Set to list #static_dofsets_
    virtual void add_dof_set_to_list() = 0;

    /// Replace a Dof Set in list #static_dofsets_ with this
    virtual void replace_in_static_dofsets(std::shared_ptr<DofSetInterface> olddofset) = 0;

    /// Get Number of Global Elements of degree of freedom row map
    virtual int num_global_elements() const = 0;

    /// Get degree of freedom row map
    virtual const Core::LinAlg::Map* dof_row_map() const = 0;

    /// Get degree of freedom column map
    virtual const Core::LinAlg::Map* dof_col_map() const = 0;

    /// Get maximum GID of degree of freedom row map
    virtual int max_all_gid() const = 0;

    /// Get minimum GID of degree of freedom row map
    virtual int min_all_gid() const = 0;

    /// Get Max of all GID assigned in the DofSets in front of current one in the list
    /// #static_dofsets_
    virtual int max_gid_in_list(MPI_Comm comm) const = 0;

    /// are the dof maps already initialized?
    virtual bool initialized() const = 0;

    //@}



    //! @name Setup and Initialization

    /// Assign dof numbers using all elements and nodes of the discretization.
    virtual int assign_degrees_of_freedom(
        const Core::FE::Discretization& dis, const unsigned dspos, const int start) = 0;

    /// reset all internal variables
    virtual void reset() = 0;

    //@}


    //! @name Proxy management
    /// Proxies need to know about changes to the DofSet.

    /// Notify proxies of new dofs
    virtual void notify_assigned() = 0;

    /// Notify proxies of reset
    virtual void notify_reset() = 0;

    /// Register new proxy to notify
    virtual void register_proxy(DofSetInterface* dofset) = 0;

    /// Remove proxy from list
    virtual void unregister(DofSetInterface* dofset) = 0;

    /// our original DofSet dies
    virtual void disconnect(DofSetInterface* dofset) = 0;

    //@}

  };  // class DofSetInterface

}  // namespace Core::DOFSets


FOUR_C_NAMESPACE_CLOSE

#endif
