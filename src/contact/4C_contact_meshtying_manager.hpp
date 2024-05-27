/*-----------------------------------------------------------------------*/
/*! \file
\level 1


\brief 4C implementation of main class to control all meshtying
*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_CONTACT_MESHTYING_MANAGER_HPP
#define FOUR_C_CONTACT_MESHTYING_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_mortar_manager_base.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace DRT
{
  class Discretization;
}

namespace CONTACT
{
  // forward declarations

  /*!
  \brief 4C implementation of main class to control all meshtying

  */
  class MtManager : public MORTAR::ManagerBase
  {
   public:
    //! @name Construction/Destruction
    //!@{

    /*!
    \brief Standard Constructor

    The constructor takes a discretization that is expected to have at least
    two meshtying boundary conditions. It extracts all meshtying boundary conditions
    and constructs one or multiple meshtying interfaces from them and stores them.

    It calls MORTAR::Interface::fill_complete() on all meshtying interfaces which
    makes the nodes and elements of a meshtying interfaces redundant on all
    processors that either own a node or an element on the interfaces in the
    input discretization.

    In addition, it creates the necessary solver strategy object which handles
    the whole meshtying evaluation.

    \param discret (in): A discretization containing meshtying boundary conditions
    \param alphaf (in): Generalized-alpha parameter (set to 0.0 by default)

    */
    MtManager(DRT::Discretization& discret, double alphaf = 0.0);



    //!@}

    //! @name Access methods
    //!@{

    //! @}

    //! @name Evaluation methods
    //!@{

    /*!
    \brief Read and check input parameters

    All specified meshtying-related input parameters are read from the
    GLOBAL::Problem::Instance() and stored into a local variable of
    type Teuchos::ParameterList. Invalid parameter combinations are
    sorted out and throw a FOUR_C_THROW.

    \param mtparams Meshtying parameter list
    \param[in] discret Underlying problem discretization

    */
    bool read_and_check_input(Teuchos::ParameterList& mtparams, const DRT::Discretization& discret);

    /*!
    \brief Write restart information for meshtying

    The additionally necessary restart information in the meshtying
    case are the current Lagrange multiplier values.

    \param[in] output IO::Discretization writer for restart
    \param forcedrestart

    */
    void WriteRestart(IO::DiscretizationWriter& output, bool forcedrestart = false) final;

    /*!
    \brief Read restart information for contact

    This method has the inverse functionality of WriteRestart, as
    it reads the restart Lagrange mulitplier vectors. Moreover,
    all mortar coupling quantities (e.g. D and M) have to be
    re-computed upon restart..

    \param reader (in): IO::Discretization reader for restart
    \param dis (in)   : global dof displacement vector
    \param zero (in)  : global dof zero vector

    */
    void read_restart(IO::DiscretizationReader& reader, Teuchos::RCP<Epetra_Vector> dis,
        Teuchos::RCP<Epetra_Vector> zero) final;

    /*!
    \brief Write interface tractions for postprocessing

    \param output (in): IO::Discretization writer for restart

    */
    void postprocess_quantities(IO::DiscretizationWriter& output) final;

    //! [derived]
    void postprocess_quantities_per_interface(
        Teuchos::RCP<Teuchos::ParameterList> outputParams) final;

    /*!
    \brief Write time step restart data/results of meshtying interfaces to output

    \param[in] outParams Parameter list with output configuration and auxiliary output data
    \param[in] writeRestart Flag to control writing of restart data
    \param[in] writeState Flag to control writing of regular result data
    */
    void OutputStep(Teuchos::RCP<Teuchos::ParameterList> outParams, const bool writeRestart,
        const bool writeState);

    //@}


   protected:
    // don't want = operator and cctor
    MtManager operator=(const MtManager& old) = delete;
    MtManager(const MtManager& old) = delete;

  };  // class MtManager
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif