// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_FLD_FLUID_XFEM_HPP
#define FOUR_C_ADAPTER_FLD_FLUID_XFEM_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_base_algorithm.hpp"
#include "4C_adapter_fld_moving_boundary.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  /// fluid with moving interfaces implemented by the XFEM
  class FluidXFEM : public FluidMovingBoundary
  {
   public:
    /// constructor
    explicit FluidXFEM(const Teuchos::ParameterList& prbdyn, std::string condname);

    /*========================================================================*/
    //! @name Misc
    /*========================================================================*/

    /// fluid field
    const std::shared_ptr<Adapter::Fluid>& fluid_field() override { return fluid_; }

    /// return the boundary discretization that matches the structure discretization
    std::shared_ptr<Core::FE::Discretization> discretization() override;

    /// return the boundary discretization that matches the structure discretization
    std::shared_ptr<Core::FE::Discretization> boundary_discretization();

    /// communication object at the interface
    std::shared_ptr<FLD::Utils::MapExtractor> const& interface() const override
    {
      return fluid_->interface();
    }

    /// communication object at the struct interface
    virtual std::shared_ptr<FLD::Utils::MapExtractor> const& struct_interface();

    //@}

    /*========================================================================*/
    //! @name Time step helpers
    /*========================================================================*/

    /// start new time step
    void prepare_time_step() override;

    /// update at time step end
    void update() override;

    /// output results
    void output() override;

    /// read restart information for given time step
    double read_restart(int step) override;

    /*========================================================================*/
    //! @name Solver calls
    /*========================================================================*/

    /// nonlinear solve
    void nonlinear_solve(std::shared_ptr<Core::LinAlg::Vector<double>> idisp,
        std::shared_ptr<Core::LinAlg::Vector<double>> ivel) override;

    /// relaxation solve
    std::shared_ptr<Core::LinAlg::Vector<double>> relaxation_solve(
        std::shared_ptr<Core::LinAlg::Vector<double>> idisp, double dt) override;
    //@}

    /*========================================================================*/
    //! @name Extract interface forces
    /*========================================================================*/

    /// After the fluid solve we need the forces at the FSI interface.
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_forces() override;
    //@}

    /*========================================================================*/
    //! @name extract helpers
    /*========================================================================*/

    /// extract the interface velocity at time t^(n+1)
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_velnp() override;

    /// extract the interface velocity at time t^n
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_veln() override;
    //@}

    /*========================================================================*/
    //! @name Number of Newton iterations
    /*========================================================================*/

    //! For simplified FD MFNK solve we want to temporally limit the
    /// number of Newton steps inside the fluid solver

    /// get the maximum number of iterations from the fluid field
    int itemax() const override { return fluid_->itemax(); }

    /// set the maximum number of iterations for the fluid field
    void set_itemax(int itemax) override { fluid_->set_itemax(itemax); }

    //@}

    /*========================================================================*/
    //! @name others
    /*========================================================================*/

    /// integrate the interface shape functions
    std::shared_ptr<Core::LinAlg::Vector<double>> integrate_interface_shape() override;

    /// create the testing of fields
    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override;



   private:
    /// fluid base algorithm object
    std::shared_ptr<Fluid> fluid_;
  };

}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
