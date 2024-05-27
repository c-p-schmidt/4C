/*----------------------------------------------------------------------*/
/*! \file
\brief Contains the functions to establish local material law
stress-strain law for isotropic material for a 3D element
following Drucker Prager plasticity model
Reference:
EA de Souza Neto, D Peric, DRJ Owen. Computational Methods of Plasticity: Theory and Applications,
John Wiley & Sons, Ltd, 2008
\level 3
*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_MAT_PLASTICDRUCKERPRAGER_HPP
#define FOUR_C_MAT_PLASTICDRUCKERPRAGER_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_inpar_material.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_local_newton.hpp"

FOUR_C_NAMESPACE_OPEN

namespace MAT
{
  namespace PAR
  {
    using FAD = Sacado::Fad::DFad<double>;
    /**
     * \brief Elasto-plasitc Drucker-Prager Material Model
     *
     * This material model simulates the elasto-plastic behaviour of materials including soil
     * and concrete based on the yield surface criteria and the plastic hardening of the material.
     *
     * Following the approach provided by EA de Souza Neto, D Peric, DRJ Owen.
     * Computational Methods of Plasticity: Theory and Applications, Page 338-339
     */
    class PlasticDruckerPrager : public CORE::MAT::PAR::Parameter
    {
     public:
      //! standard constructor
      PlasticDruckerPrager(Teuchos::RCP<CORE::MAT::PAR::Material> matdata);
      //! @name material parameters
      //@{
      //! Young's modulus
      const double youngs_;
      //! Possion's ratio
      const double poissonratio_;
      //! Density
      const double density_;
      //! linear isotropic hardening modulus
      const double isohard_;
      //! tolerance for local Newton iteration
      const double abstol_;
      //! initial cohesion
      const double cohesion_;
      //! Mohr-Coulumb approximation parameters
      const double eta_;
      const double xi_;
      //! dilatancy angle variable
      const double etabar_;
      const int itermax_;
      //@}
      //! create material instance of matching type with my parameters
      Teuchos::RCP<CORE::MAT::Material> CreateMaterial() override;
    };
  }  // namespace PAR
  class PlasticDruckerPragerType : public CORE::COMM::ParObjectType
  {
   public:
    std::string Name() const override { return "PlasticDruckerPragerType"; }
    static PlasticDruckerPragerType& Instance() { return instance_; };
    CORE::COMM::ParObject* Create(const std::vector<char>& data) override;

   private:
    static PlasticDruckerPragerType instance_;
  };

  class PlasticDruckerPrager : public So3Material
  {
   public:
    PlasticDruckerPrager();

    explicit PlasticDruckerPrager(MAT::PAR::PlasticDruckerPrager* params);
    int UniqueParObjectId() const override
    {
      return PlasticDruckerPragerType::Instance().UniqueParObjectId();
    }
    void Pack(CORE::COMM::PackBuffer& data) const override;
    void Unpack(const std::vector<char>& data) override;
    CORE::Materials::MaterialType MaterialType() const override
    {
      return CORE::Materials::m_pldruckprag;
    }
    void ValidKinematics(INPAR::STR::KinemType kinem) override
    {
      if (kinem != INPAR::STR::KinemType::linear)
        FOUR_C_THROW(
            "The plastic Drucker Prager material model is only compatible with linear kinematics.");
    }
    Teuchos::RCP<CORE::MAT::Material> Clone() const override
    {
      return Teuchos::rcp(new PlasticDruckerPrager(*this));
    }
    void Setup(int numgp, INPUT::LineDefinition* linedef) override;
    void Update() override;
    /**
     * \brief Evaulate the stresses from the strains in the material
     *
     * \param defgrad :deformation gradient
     * \param linstrain :linear total strains
     * \param params :parameter list for communication
     * \param stress :2nd PK-stress
     * \param cmat :material stiffness matrix
     * \param gp :Gauss point
     * \param eleGID :element global identifier
     */
    void Evaluate(const CORE::LINALG::Matrix<3, 3>* defgrd,
        const CORE::LINALG::Matrix<NUM_STRESS_3D, 1>* linstrain, Teuchos::ParameterList& params,
        CORE::LINALG::Matrix<NUM_STRESS_3D, 1>* stress,
        CORE::LINALG::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat, int gp, int eleGID) override
    {
      this->EvaluateFAD(defgrd, linstrain, params, stress, cmat, gp, eleGID);
    };

    template <typename ScalarT>
    void Evaluate(const CORE::LINALG::Matrix<3, 3>* defgrd,
        const CORE::LINALG::Matrix<NUM_STRESS_3D, 1, ScalarT>* linstrain,
        Teuchos::ParameterList& params, CORE::LINALG::Matrix<NUM_STRESS_3D, 1, ScalarT>* stress,
        CORE::LINALG::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat, int gp, int eleGID)
    {
      this->EvaluateFAD(defgrd, linstrain, params, stress, cmat, gp, eleGID);
    };
    template <typename ScalarT>
    void EvaluateFAD(const CORE::LINALG::Matrix<3, 3>* defgrd,
        const CORE::LINALG::Matrix<NUM_STRESS_3D, 1, ScalarT>* linstrain,
        Teuchos::ParameterList& params, CORE::LINALG::Matrix<NUM_STRESS_3D, 1, ScalarT>* stress,
        CORE::LINALG::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat, int gp, int eleGID);
    template <typename T>
    void Stress(const T p, const CORE::LINALG::Matrix<NUM_STRESS_3D, 1, T>& devstress,
        CORE::LINALG::Matrix<NUM_STRESS_3D, 1, T>& stress);

    void setup_cmat(CORE::LINALG::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>& cmat);
    /**
     * \brief setup the elastoplasticity tensor in matrix notation for 3d return to cone
     *
     * \param cmat :elasto-plastic tangent modulus (out)
     * \param Dgamma :plastic multiplier
     * \param G :shear modulus
     * \param Kappa :Bulk modulus
     * \param devstrain :deviatoric strain
     * \param xi :Mohr-columb parameter
     * \param Hiso :isotropic hardening modulus
     * \param eta :Mohr-columb parameter
     * \param etabar :Mohr-columb parameter
     */
    void setup_cmat_elasto_plastic_cone(CORE::LINALG::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>& cmat,
        double Dgamma, double G, double Kappa, CORE::LINALG::Matrix<NUM_STRESS_3D, 1>& devstrain,
        double xi, double Hiso, double eta, double etabar);
    /**
     * \brief setup the elastoplasticity tensor in matrix notation for 3d return to apex
     *
     * \param cmat :elasto-plastic tangent modulus (out)
     * \param Kappa :Bulk modulus
     * \param devstrain :deviatoric strain
     * \param xi :Mohr-columb parameter
     * \param Hiso :isotropic hardening modulus
     * \param eta :Mohr-columb parameter
     * \param etabar :Mohr-columb parameter
     */
    void setup_cmat_elasto_plastic_apex(CORE::LINALG::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
                                            cmat,           // elasto-plastic tangent modulus (out)
        double Kappa,                                       // Bulk mmodulus
        CORE::LINALG::Matrix<NUM_STRESS_3D, 1>& devstrain,  // deviatoric strain
        double xi,                                          // Mohr-columb parameter
        double Hiso,                                        // isotropic hardening modulus
        double eta,                                         // Mohr-columb parameter
        double etabar                                       // Mohr-columb parameter
    );
    CORE::MAT::PAR::Parameter* Parameter() const override { return params_; }
    double Density() const override { return params_->density_; }
    template <typename T>
    std::pair<T, T> return_to_cone_funct_and_deriv(T Dgamma, T G, T kappa, T Phi_trial);
    template <typename T>
    std::pair<T, T> return_to_apex_funct_and_deriv(T dstrainv, T p, T kappa, T strainbar_p);
    bool Initialized() const { return (isinit_ and !strainplcurr_.empty()); }
    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool EvaluateOutputData(
        const std::string& name, CORE::LINALG::SerialDenseMatrix& data) const override;

   private:
    MAT::PAR::PlasticDruckerPrager* params_;
    std::vector<CORE::LINALG::Matrix<NUM_STRESS_3D, 1>> strainpllast_;
    std::vector<CORE::LINALG::Matrix<NUM_STRESS_3D, 1>> strainplcurr_;
    std::vector<double> strainbarpllast_;
    std::vector<double> strainbarplcurr_;
    bool isinit_;
  };
}  // namespace MAT
FOUR_C_NAMESPACE_CLOSE

#endif