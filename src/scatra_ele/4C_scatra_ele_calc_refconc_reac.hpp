/*----------------------------------------------------------------------*/
/*! \file
\brief main file containing routines for calculation of scatra element formulated in reference
concentrations and with advanced reaction terms

\level 3

 *----------------------------------------------------------------------*/


#ifndef FOUR_C_SCATRA_ELE_CALC_REFCONC_REAC_HPP
#define FOUR_C_SCATRA_ELE_CALC_REFCONC_REAC_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_calc_advanced_reaction.hpp"

FOUR_C_NAMESPACE_OPEN


namespace DRT
{
  namespace ELEMENTS
  {
    template <CORE::FE::CellType distype>
    class ScaTraEleCalcRefConcReac : public ScaTraEleCalcAdvReac<distype>
    {
     private:
      /// private constructor, since we are a Singleton.
      ScaTraEleCalcRefConcReac(
          const int numdofpernode, const int numscal, const std::string& disname);

      typedef ScaTraEleCalc<distype> my;
      typedef ScaTraEleCalcAdvReac<distype> advreac;
      using my::nen_;
      using my::nsd_;

     public:
      /// Singleton access method
      static ScaTraEleCalcRefConcReac<distype>* Instance(
          const int numdofpernode, const int numscal, const std::string& disname);

     protected:
      //! Set reac. body force, reaction coefficient and derivatives
      void set_advanced_reaction_terms(const int k,               //!< index of current scalar
          const Teuchos::RCP<MAT::MatListReactions> matreaclist,  //!< index of current scalar
          const double* gpcoord  //!< current Gauss-point coordinates
          ) override;

      //! calculation of convective element matrix: add conservative contributions
      void calc_mat_conv_add_cons(
          CORE::LINALG::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                            //!< index of current scalar
          const double timefacfac,  //!< domain-integration factor times time-integration factor
          const double vdiv,        //!< velocity divergence
          const double densnp       //!< density at time_(n+1)
          ) override;

      //! set internal variables
      void set_internal_variables_for_mat_and_rhs() override;

      //! calculation of diffusive element matrix
      void CalcMatDiff(CORE::LINALG::SerialDenseMatrix& emat,  //!< element matrix to be filled
          const int k,                                         //!< index of current scalar
          const double timefacfac  //!< domain-integration factor times time-integration factor
          ) override;

      //! calculate the Laplacian (weak form)
      void get_laplacian_weak_form(double& val,                //!< ?
          const CORE::LINALG::Matrix<nsd_, nsd_>& difftensor,  //!< ?
          const int vi,                                        //!< ?
          const int ui                                         //!< ?
      )
      {
        val = 0.0;
        for (unsigned j = 0; j < nsd_; j++)
        {
          for (unsigned i = 0; i < nsd_; i++)
          {
            val += my::derxy_(j, vi) * difftensor(j, i) * my::derxy_(i, ui);
          }
        }
        return;
      };

      //! standard Galerkin diffusive term on right hand side
      void calc_rhs_diff(CORE::LINALG::SerialDenseVector& erhs,  //!< element vector to be filled
          const int k,                                           //!< index of current scalar
          const double rhsfac  //!< time-integration factor for rhs times domain-integration factor
          ) override;

      //! calculate the Laplacian (weak form)
      void get_laplacian_weak_form_rhs(double& val,            //!< ?
          const CORE::LINALG::Matrix<nsd_, nsd_>& difftensor,  //!< ?
          const CORE::LINALG::Matrix<nsd_, 1>& gradphi,        //!< ?
          const int vi                                         //!< ?
      )
      {
        val = 0.0;
        for (unsigned j = 0; j < nsd_; j++)
        {
          for (unsigned i = 0; i < nsd_; i++)
          {
            val += my::derxy_(j, vi) * difftensor(j, i) * gradphi(i);
          }
        }
        return;
      };

      // add nodal displacements to point coordinates
      void update_node_coordinates() override
      { /*nothing to to since we want reference coordinates*/
        return;
      };

     private:
      /// determinante of deformation gradient
      double j_;

      /// inverse of cauchy-green deformation gradient
      CORE::LINALG::Matrix<nsd_, nsd_> c_inv_;

      /// derivative dJ/dX by finite differences
      CORE::LINALG::Matrix<nsd_, 1> d_jd_x_;

    };  // end ScaTraEleCalcRefConcReac


  }  // namespace ELEMENTS

}  // namespace DRT


FOUR_C_NAMESPACE_CLOSE

#endif