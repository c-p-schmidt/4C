/*----------------------------------------------------------------------*/
/*! \file

\brief Basis of all WEAR algorithms that perform a coupling between the
       structural field equation and ALE field equations

\level 2

*/
/*----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | definitions                                              farah 11/13 |
 *----------------------------------------------------------------------*/
#ifndef FOUR_C_WEAR_ALGORITHM_HPP
#define FOUR_C_WEAR_ALGORITHM_HPP


/*----------------------------------------------------------------------*
 | headers                                                  farah 11/13 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"

#include <Epetra_Vector.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | forward declarations                                     farah 11/13 |
 *----------------------------------------------------------------------*/
namespace DRT
{
  class Discretization;
}

namespace ADAPTER
{
  class Structure;
  class FSIStructureWrapper;
  class AleWearWrapper;
}  // namespace ADAPTER

namespace ALE
{
  class Ale;
}

namespace MORTAR
{
  class ManagerBase;
}

namespace CONTACT
{
  class Interface;
}

/*----------------------------------------------------------------------*
 |                                                          farah 11/13 |
 *----------------------------------------------------------------------*/
namespace WEAR
{
  class Algorithm : public ADAPTER::AlgorithmBase
  {
   public:
    //! create using a Epetra_Comm
    explicit Algorithm(const Epetra_Comm& comm);


    //! outer level time loop (to be implemented by deriving classes)
    virtual void TimeLoop() = 0;

    //! read restart data
    void read_restart(int step  //!< step number where the calculation is continued
        ) override = 0;

    //! access to structural field
    Teuchos::RCP<ADAPTER::FSIStructureWrapper> StructureField() { return structure_; }

    //! access to ALE field
    ADAPTER::AleWearWrapper& ale_field() { return *ale_; }

   private:
    //! check compatibility if input parameters
    void check_input();

    //! create mortar interfaces for material conf.
    void create_material_interface();

    //! @name Underlying fields
    Teuchos::RCP<ADAPTER::FSIStructureWrapper> structure_;  //! underlying structure
    Teuchos::RCP<ADAPTER::AleWearWrapper> ale_;             //! underlying ALE

   protected:
    int dim_;  //! problem dimension
    //@}

    Teuchos::RCP<MORTAR::ManagerBase> cmtman_;                     // contact manager
    std::vector<Teuchos::RCP<CONTACT::Interface>> interfaces_;     // contact/wear interfaces
    std::vector<Teuchos::RCP<CONTACT::Interface>> interfacesMat_;  // contact interfaces in Mat.

  };  // Algorithm
}  // namespace WEAR


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif