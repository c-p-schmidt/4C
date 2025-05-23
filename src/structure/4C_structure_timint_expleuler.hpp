// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_TIMINT_EXPLEULER_HPP
#define FOUR_C_STRUCTURE_TIMINT_EXPLEULER_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_structure_timint_expl.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* belongs to structural dynamics namespace */
namespace Solid
{
  /*====================================================================*/
  /*!
   * \brief forward Euler: 1st order accurate,
   *                       explicit time integrator,

   */
  class TimIntExplEuler : public TimIntExpl
  {
   public:
    //! @name Life
    //@{

    //! Constructor
    TimIntExplEuler(const Teuchos::ParameterList& timeparams,  //!< time params
        const Teuchos::ParameterList& ioparams,                //!< ioflags
        const Teuchos::ParameterList& sdynparams,              //!< input parameters
        const Teuchos::ParameterList& xparams,                 //!< extra flags
        std::shared_ptr<Core::FE::Discretization> actdis,      //!< current discretisation
        std::shared_ptr<Core::LinAlg::Solver> solver,          //!< the solver
        std::shared_ptr<Core::LinAlg::Solver> contactsolver,   //!< the solver for contact meshtying
        std::shared_ptr<Core::IO::DiscretizationWriter> output  //!< the output
    );

    //! Copy constructor
    TimIntExplEuler(const TimIntExplEuler& old) : TimIntExpl(old) { ; }

    /*! \brief Initialize this object

    Hand in all objects/parameters/etc. from outside.
    Construct and manipulate internal objects.

    \note Try to only perform actions in init(), which are still valid
          after parallel redistribution of discretizations.
          If you have to perform an action depending on the parallel
          distribution, make sure you adapt the affected objects after
          parallel redistribution.
          Example: cloning a discretization from another discretization is
          OK in init(...). However, after redistribution of the source
          discretization do not forget to also redistribute the cloned
          discretization.
          All objects relying on the parallel distribution are supposed to
          the constructed in \ref setup().

    \warning none
    \return bool

    */
    void init(const Teuchos::ParameterList& timeparams, const Teuchos::ParameterList& sdynparams,
        const Teuchos::ParameterList& xparams, std::shared_ptr<Core::FE::Discretization> actdis,
        std::shared_ptr<Core::LinAlg::Solver> solver) override;

    /*! \brief Setup all class internal objects and members

     setup() is not supposed to have any input arguments !

     Must only be called after init().

     Construct all objects depending on the parallel distribution and
     relying on valid maps like, e.g. the state vectors, system matrices, etc.

     Call all setup() routines on previously initialized internal objects and members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, e.g. vectors may have wrong maps.

    \warning none
    \return void

    */
    void setup() override;

    //@}

    //! @name Actions
    //@{

    //! Resize \p TimIntMStep<T> multi-step quantities
    void resize_m_step() override;

    //! Do time integration of single step
    int integrate_step() override;

    //! Update configuration after time step
    //!
    //! Thus the 'last' converged is lost and a reset of the time step
    //! becomes impossible. We are ready and keen awaiting the next time step.
    void update_step_state() override;

    //! Update Element
    void update_step_element() override;

    //@}

    //! @name Attribute access functions
    //@{

    //! Return time integrator name
    enum Inpar::Solid::DynamicType method_name() const override
    {
      return Inpar::Solid::DynamicType::ExplEuler;
    }

    //! Provide number of steps, e.g. a single-step method returns 1,
    //! a m-multistep method returns m
    int method_steps() const override { return 1; }

    //! Give local order of accuracy of displacement part
    int method_order_of_accuracy_dis() const override { return 1; }

    //! Give local order of accuracy of velocity part
    int method_order_of_accuracy_vel() const override { return 1; }

    /*! \brief Return linear error coefficient of displacements
     *
     *  The local discretization error reads
     *  \f[
     *  e \approx \frac{1}{2}\Delta t_n^2 \ddot{d_n} + HOT(\Delta t_n^3)
     *  \f]
     */
    double method_lin_err_coeff_dis() const override { return 0.5; }

    /*! \brief Return linear error coefficient of velocities
     *
     *  The local discretization error reads
     *  \f[
     *  e \approx \frac{1}{2}\Delta t_n^2 \dddot{d_n} + HOT(\Delta t_n^3)
     *  \f]
     */
    double method_lin_err_coeff_vel() const override { return 0.5; }

    //@}

    //! @name System vectors
    //@{

    //! Return external force \f$F_{ext,n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fext() override { return fextn_; }

    //! Return external force \f$F_{ext,n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fext_new() override
    {
      FOUR_C_THROW("FextNew() not available in AB2");
      return nullptr;
    }

    //! Read and set restart for forces
    void read_restart_force() override;

    //! Write internal and external forces for restart
    void write_restart_force(std::shared_ptr<Core::IO::DiscretizationWriter> output) override;

    //@}


   protected:
    bool modexpleuler_;  //!< modified explicit Euler equation (veln_ instead of vel_ for calc of
                         //!< disn_), default: true

    //! @name Global forces at \f$t_{n+1}\f$
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> fextn_;   //!< external force
                                                            //!< \f$F_{int;n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fintn_;   //!< internal force
                                                            //!< \f$F_{int;n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fviscn_;  //!< Rayleigh viscous forces
                                                            //!< \f$C \cdot V_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fcmtn_;   //!< contact or meshtying forces
                                                            //!< \f$F_{cmt;n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>>
        frimpn_;  //!< time derivative of
                  //!< linear momentum
                  //!< (temporal rate of impulse)
                  //!< \f$\dot{P}_{n+1} = M \cdot \dot{V}_{n+1}\f$
    //@}

  };  // class TimIntExplEuler

}  // namespace Solid

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
