/*----------------------------------------------------------------------*/
/*! \file


 \brief coupling adapter for porous meshtying

// Masterthesis of h.Willmann under supervision of Anh-Tu Vuong and Johannes Kremheller
// Originates from ADAPTER::CouplingNonLinMortar

\level 3

*----------------------------------------------------------------------*/

#ifndef FOUR_C_ADAPTER_COUPLING_PORO_MORTAR_HPP
#define FOUR_C_ADAPTER_COUPLING_PORO_MORTAR_HPP

/*---------------------------------------------------------------------*
 | headers                                                  ager 10/15 |
 *---------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_adapter_coupling_nonlin_mortar.hpp"
#include "4C_contact_lagrange_strategy_poro.hpp"
#include "4C_coupling_adapter.hpp"

#include <Epetra_Comm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN
/*---------------------------------------------------------------------*
 | forward declarations                                     ager 10/15 |
 *---------------------------------------------------------------------*/
namespace DRT
{
  class Discretization;
}  // namespace DRT

namespace CONTACT
{
  class Interface;
}

namespace CORE::LINALG
{
  class SparseMatrix;
}

namespace ADAPTER
{
  class CouplingPoroMortar : public CouplingNonLinMortar
  {
   public:
    /*!
    \brief Empty constructor

    */
    CouplingPoroMortar(int spatial_dimension, Teuchos::ParameterList mortar_coupling_params,
        Teuchos::ParameterList contact_dynamic_params,
        CORE::FE::ShapeFunctionType shape_function_type);


    virtual void EvaluatePoroMt(Teuchos::RCP<Epetra_Vector> fvel, Teuchos::RCP<Epetra_Vector> svel,
        Teuchos::RCP<Epetra_Vector> fpres, Teuchos::RCP<Epetra_Vector> sdisp,
        const Teuchos::RCP<DRT::Discretization> sdis, Teuchos::RCP<CORE::LINALG::SparseMatrix>& f,
        Teuchos::RCP<CORE::LINALG::SparseMatrix>& k_fs, Teuchos::RCP<Epetra_Vector>& frhs,
        CORE::ADAPTER::Coupling& coupfs, Teuchos::RCP<const Epetra_Map> fdofrowmap);

    void UpdatePoroMt();

    void recover_fluid_lm_poro_mt(
        Teuchos::RCP<Epetra_Vector> disi, Teuchos::RCP<Epetra_Vector> veli);  // h.Willmann

    // return the used poro lagrange strategy
    Teuchos::RCP<CONTACT::LagrangeStrategyPoro> GetPoroStrategy()
    {
      if (porolagstrategy_ == Teuchos::null) FOUR_C_THROW("GetPoroStrategy(): No strategy set!");
      return porolagstrategy_;
    };

   protected:
    /*!
    \brief Read Mortar Condition

    */
    void ReadMortarCondition(Teuchos::RCP<DRT::Discretization> masterdis,
        Teuchos::RCP<DRT::Discretization> slavedis, std::vector<int> coupleddof,
        const std::string& couplingcond, Teuchos::ParameterList& input,
        std::map<int, DRT::Node*>& mastergnodes, std::map<int, DRT::Node*>& slavegnodes,
        std::map<int, Teuchos::RCP<DRT::Element>>& masterelements,
        std::map<int, Teuchos::RCP<DRT::Element>>& slaveelements) override;

    /*!
    \brief Add Mortar Elments

    */
    void AddMortarElements(Teuchos::RCP<DRT::Discretization> masterdis,
        Teuchos::RCP<DRT::Discretization> slavedis, Teuchos::ParameterList& input,
        std::map<int, Teuchos::RCP<DRT::Element>>& masterelements,
        std::map<int, Teuchos::RCP<DRT::Element>>& slaveelements,
        Teuchos::RCP<CONTACT::Interface>& interface, int numcoupleddof) override;

    /*!
    \brief complete interface, store as internal variable
           store maps as internal variable and do parallel redist.

    */
    void CompleteInterface(Teuchos::RCP<DRT::Discretization> masterdis,
        Teuchos::RCP<CONTACT::Interface>& interface) override;

    /*!
    \brief create strategy object if required

    */
    void CreateStrategy(Teuchos::RCP<DRT::Discretization> masterdis,
        Teuchos::RCP<DRT::Discretization> slavedis, Teuchos::ParameterList& input,
        int numcoupleddof) override;

   private:
    // poro lagrange strategy
    Teuchos::RCP<CONTACT::LagrangeStrategyPoro> porolagstrategy_;

    // firstinit
    bool firstinit_;

    int slavetype_;   // 1 poro, 0 struct, -1 default
    int mastertype_;  // 1 poro, 0 struct, -1 default
  };
}  // namespace ADAPTER

FOUR_C_NAMESPACE_CLOSE

#endif