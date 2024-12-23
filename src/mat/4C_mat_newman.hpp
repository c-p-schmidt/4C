// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_NEWMAN_HPP
#define FOUR_C_MAT_NEWMAN_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_elchsinglemat.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for convection-diffusion
    class Newman : public ElchSingleMat
    {
     public:
      /// standard constructor
      Newman(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      /// valence (= charge number)
      const double valence_;

      /// definition of transference number
      /// (by curve number or implemented concentration dependence)
      const int transnrcurve_;

      /// definition of thermodynamic factor
      /// (by curve number or implemented concentration dependence)
      const int thermfaccurve_;

      // Important:
      // pointer to vectors containing parameter for predefined curves
      // -> faster than saving the actual vector

      /// number of parameter needed for implemented concentration dependence
      const int transnrparanum_;
      /// parameter needed for implemented concentration dependence
      const std::vector<double> transnrpara_;

      /// number of parameter needed for implemented concentration dependence
      const int thermfacparanum_;
      /// parameter needed for implemented concentration dependence
      const std::vector<double> thermfacpara_;
      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;
    };  // class Newman

  }  // namespace PAR

  class NewmanType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "NewmanType"; }

    static NewmanType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static NewmanType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for the material properties of an ion species in an electrolyte solution
  class Newman : public ElchSingleMat
  {
   public:
    /// construct empty material object
    Newman();

    /// construct the material object given material parameters
    explicit Newman(Mat::PAR::Newman* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return NewmanType::instance().unique_par_object_id();
    }

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique parobject id delivered by unique_par_object_id() which will then
      identify the exact class on the receiving processor.

      \param data (in/out): char vector to store class information
    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
      \brief Unpack data from a char vector into this class

      The vector data contains all information to rebuild the
      exact copy of an instance of a class on a different processor.
      The first entry in data has to be an integer which is the unique
      parobject id defined at the top of this file and delivered by
      unique_par_object_id().

      \param data (in) : vector storing all data to be unpacked into this
      instance.
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_newman;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<Newman>(*this);
    }

    /// valence (= charge number)
    double valence() const { return params_->valence_; }

    /// computation of the transference number based on the defined curve
    double compute_transference_number(const double cint) const;
    /// computation of the first derivative of the transference number based on the defined curve
    double compute_first_deriv_trans(const double cint) const;

    /// computation of the thermodynamic factor based on the defined curve
    double compute_therm_fac(const double cint) const;
    /// computation of the first derivative of the transference number based on the defined curve
    double compute_first_deriv_therm_fac(const double cint) const;

   private:
    /// return curve defining the transference number
    int trans_nr_curve() const { return params_->transnrcurve_; }
    /// return curve defining the thermodynamic factor
    int therm_fac_curve() const { return params_->thermfaccurve_; }

    /// parameter needed for implemented concentration dependence
    const std::vector<double>& trans_nr_params() const { return params_->transnrpara_; }

    /// parameter needed for implemented concentration dependence
    const std::vector<double>& therm_fac_params() const { return params_->thermfacpara_; }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// my material parameters
    Mat::PAR::Newman* params_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
