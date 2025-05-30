// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_TURBULENCE_TRANSFER_TURB_INFLOW_HPP
#define FOUR_C_FLUID_TURBULENCE_TRANSFER_TURB_INFLOW_HPP

#include "4C_config.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::Conditions
{
  class Condition;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace FLD
{
  class TransferTurbulentInflowCondition
  {
   public:
    /*!
    \brief Standard Constructor

    */
    TransferTurbulentInflowCondition(std::shared_ptr<Core::FE::Discretization> dis,
        std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps);

    /*!
    \brief Destructor

    */
    virtual ~TransferTurbulentInflowCondition() = default;

    /*!
    \brief Transfer process copying values from master boundary to slave
           boundary (slave must be of Dirichlet type, otherwise this
           operation doesn't make to mauch sense)

           Intended to be called after ApplyDirichlet, overwriting the
           dummy Dirichlet values on the slave boundary by the values
           of the last time step on the master boundary
    */
    virtual void transfer(const std::shared_ptr<Core::LinAlg::Vector<double>> veln,
        std::shared_ptr<Core::LinAlg::Vector<double>> velnp, const double time);

   protected:
    //! there are two types of transfer conditions. values are transferred
    //! from master to slave conditions
    enum ToggleType
    {
      none,
      master,
      slave
    };

    //! get all data from condition
    void get_data(int& id, int& direction, ToggleType& type, const Core::Conditions::Condition*);

    //! receive a block in the round robin communication pattern
    void receive_block(
        std::vector<char>& rblock, Core::Communication::Exporter& exporter, MPI_Request& request);

    //! send a block in the round robin communication pattern
    void send_block(
        std::vector<char>& sblock, Core::Communication::Exporter& exporter, MPI_Request& request);

    //! unpack all master values contained in receive block
    void unpack_local_master_values(std::vector<int>& mymasters,
        std::vector<std::vector<double>>& mymasters_vel, std::vector<char>& rblock);

    //! pack all master values into a send block
    void pack_local_master_values(std::vector<int>& mymasters,
        std::vector<std::vector<double>>& mymasters_vel, Core::Communication::PackBuffer& sblock);

    //! for all values available on the processor, do the final setting of the value
    virtual void set_values_available_on_this_proc(std::vector<int>& mymasters,
        std::vector<std::vector<double>>& mymasters_vel,
        std::shared_ptr<Core::LinAlg::Vector<double>> velnp);

    //! flag active boundary condition (may be used to switch off everything)
    bool active_;

    //! the discretisation
    std::shared_ptr<Core::FE::Discretization> dis_;

    //! info on DIirchlet boundary
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps_;

    //! the connectivity of the boundary condition
    std::map<int, std::vector<int>> midtosid_;

    //! time curve number
    int curve_;

    int numveldof_;
  };

  class TransferTurbulentInflowConditionXW : public TransferTurbulentInflowCondition
  {
   public:
    /*!
    \brief Standard Constructor

    */
    TransferTurbulentInflowConditionXW(std::shared_ptr<Core::FE::Discretization> dis,
        std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps);


    /*!
    \brief Transfer process copying values from master boundary to slave
           boundary (slave must be of Dirichlet type, otherwise this
           operation doesn't make to mauch sense)

           Intended to be called after ApplyDirichlet, overwriting the
           dummy Dirichlet values on the slave boundary by the values
           of the last time step on the master boundary
    */
    void transfer(const std::shared_ptr<Core::LinAlg::Vector<double>> veln,
        std::shared_ptr<Core::LinAlg::Vector<double>> velnp, const double time) override;

   private:
    //! for all values available on the processor, do the final setting of the value
    void set_values_available_on_this_proc(std::vector<int>& mymasters,
        std::vector<std::vector<double>>& mymasters_vel,
        std::shared_ptr<Core::LinAlg::Vector<double>> velnp) override;
  };


  class TransferTurbulentInflowConditionNodal : public TransferTurbulentInflowCondition
  {
   public:
    /*!
    \brief Standard Constructor

    */
    TransferTurbulentInflowConditionNodal(std::shared_ptr<Core::FE::Discretization> dis,
        std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps);


    /*!
    \brief Transfer process copying values from master boundary to slave
           boundary (slave must be of Dirichlet type, otherwise this
           operation doesn't make to mauch sense)

           Intended to be called after ApplyDirichlet, overwriting the
           dummy Dirichlet values on the slave boundary by the values
           of the last time step on the master boundary
    */
    void transfer(const std::shared_ptr<Core::LinAlg::Vector<double>> invec,
        std::shared_ptr<Core::LinAlg::Vector<double>> outvec, const double time) override;

    bool is_active() { return active_; }

   private:
    //! for all values available on the processor, do the final setting of the value
    void set_values_available_on_this_proc(std::vector<int>& mymasters,
        std::vector<std::vector<double>>& mymasters_vec,
        std::shared_ptr<Core::LinAlg::Vector<double>> outvec) override;
  };

}  // end namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
