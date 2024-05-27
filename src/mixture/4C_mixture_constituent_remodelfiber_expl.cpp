/*----------------------------------------------------------------------*/
/*! \file
\brief Definition of a remodel constituent with explicit integration of the local evolution
equations
\level 3
*/
/*----------------------------------------------------------------------*/
#include "4C_mixture_constituent_remodelfiber_expl.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_matelast_aniso_structuraltensor_strategy.hpp"
#include "4C_mixture_constituent.hpp"
#include "4C_mixture_constituent_remodelfiber_lib.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential_active.hpp"
#include "4C_mixture_growth_evolution_linear_cauchy_poisson_turnover.hpp"
#include "4C_utils_function_of_time.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>

FOUR_C_NAMESPACE_OPEN

// anonymous namespace for helper classes and functions
namespace
{
  [[nodiscard]] CORE::LINALG::Matrix<3, 3> EvaluateC(const CORE::LINALG::Matrix<3, 3>& F)
  {
    CORE::LINALG::Matrix<3, 3> C(false);
    C.MultiplyTN(F, F);
    return C;
  }

  [[nodiscard]] CORE::LINALG::Matrix<3, 3> EvaluateiCext(const CORE::LINALG::Matrix<3, 3>& iFext)
  {
    CORE::LINALG::Matrix<3, 3> iCext(false);
    iCext.MultiplyNT(iFext, iFext);
    return iCext;
  }
}  // namespace

MIXTURE::PAR::MixtureConstituentRemodelFiberExpl::MixtureConstituentRemodelFiberExpl(
    const Teuchos::RCP<CORE::MAT::PAR::Material>& matdata)
    : MixtureConstituent(matdata),
      fiber_id_(matdata->Get<int>("FIBER_ID") - 1),
      init_(matdata->Get<int>("INIT")),
      gamma_(matdata->Get<double>("GAMMA")),
      fiber_material_id_(matdata->Get<int>("FIBER_MATERIAL_ID")),
      fiber_material_(FiberMaterialFactory(fiber_material_id_)),
      growth_enabled_(matdata->Get<bool>("GROWTH_ENABLED")),
      poisson_decay_time_(matdata->Get<double>("DECAY_TIME")),
      growth_constant_(matdata->Get<double>("GROWTH_CONSTANT")),
      deposition_stretch_(matdata->Get<double>("DEPOSITION_STRETCH")),
      deposition_stretch_timefunc_num_(matdata->Get<int>("DEPOSITION_STRETCH_TIMEFUNCT")),
      inelastic_external_deformation_(matdata->Get<bool>("INELASTIC_GROWTH"))
{
}

std::unique_ptr<MIXTURE::MixtureConstituent>
MIXTURE::PAR::MixtureConstituentRemodelFiberExpl::CreateConstituent(int id)
{
  return std::make_unique<MIXTURE::MixtureConstituentRemodelFiberExpl>(this, id);
}

MIXTURE::MixtureConstituentRemodelFiberExpl::MixtureConstituentRemodelFiberExpl(
    MIXTURE::PAR::MixtureConstituentRemodelFiberExpl* params, int id)
    : MixtureConstituent(params, id),
      params_(params),
      remodel_fiber_(),
      anisotropy_extension_(params_->init_, params_->gamma_, false,
          Teuchos::rcp(new MAT::ELASTIC::StructuralTensorStrategyStandard(nullptr)),
          {params_->fiber_id_})
{
  anisotropy_extension_.register_needed_tensors(
      MAT::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR_STRESS |
      MAT::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
}

CORE::Materials::MaterialType MIXTURE::MixtureConstituentRemodelFiberExpl::MaterialType() const
{
  return CORE::Materials::mix_remodelfiber_expl;
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::PackConstituent(
    CORE::COMM::PackBuffer& data) const
{
  MIXTURE::MixtureConstituent::PackConstituent(data);
  anisotropy_extension_.PackAnisotropy(data);

  for (const RemodelFiber<2>& fiber : remodel_fiber_) fiber.Pack(data);
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::UnpackConstituent(
    std::vector<char>::size_type& position, const std::vector<char>& data)
{
  MIXTURE::MixtureConstituent::UnpackConstituent(position, data);
  initialize();

  anisotropy_extension_.UnpackAnisotropy(data, position);
  for (RemodelFiber<2>& fiber : remodel_fiber_) fiber.Unpack(position, data);
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::register_anisotropy_extensions(
    MAT::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::initialize()
{
  std::shared_ptr<const RemodelFiberMaterial<double>> material =
      params_->fiber_material_->create_remodel_fiber_material();

  remodel_fiber_.clear();
  for (int gp = 0; gp < num_gp(); ++gp)
  {
    LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<double> growth_evolution(
        params_->growth_constant_, params_->poisson_decay_time_);
    remodel_fiber_.emplace_back(material, growth_evolution, evaluate_deposition_stretch(0.0));
  }
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::ReadElement(
    int numgp, INPUT::LineDefinition* linedef)
{
  MIXTURE::MixtureConstituent::ReadElement(numgp, linedef);
  initialize();
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::Setup(Teuchos::ParameterList& params, int eleGID)
{
  MIXTURE::MixtureConstituent::Setup(params, eleGID);
  update_homeostatic_values(params, eleGID);
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::UpdateElasticPart(
    const CORE::LINALG::Matrix<3, 3>& F, const CORE::LINALG::Matrix<3, 3>& iFext,
    Teuchos::ParameterList& params, const double dt, const int gp, const int eleGID)
{
  MixtureConstituent::UpdateElasticPart(F, iFext, params, dt, gp, eleGID);

  if (!params_->inelastic_external_deformation_)
  {
    FOUR_C_THROW(
        "You specified that there is no inelastic external deformation in the input file, but this "
        "method is only called if there is one. Probably, you are using a mixture rule with "
        "inelastic growth. You have to set INELASTIC_GROWTH to true or use a different growth "
        "rule.");
  }
  const double lambda_f = evaluate_lambdaf(EvaluateC(F), gp, eleGID);
  const double lambda_ext = evaluate_lambda_ext(iFext, gp, eleGID);
  remodel_fiber_[gp].set_state(lambda_f, lambda_ext);
  remodel_fiber_[gp].Update();

  if (params_->growth_enabled_)
  {
    const double dt = params.get<double>("delta time");

    remodel_fiber_[gp].integrate_local_evolution_equations_explicit(dt);
  }
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::Update(const CORE::LINALG::Matrix<3, 3>& F,
    Teuchos::ParameterList& params, const int gp, const int eleGID)
{
  MixtureConstituent::Update(F, params, gp, eleGID);

  update_homeostatic_values(params, eleGID);

  if (!params_->inelastic_external_deformation_)
  {
    // Update state
    const double lambda_f = evaluate_lambdaf(EvaluateC(F), gp, eleGID);
    remodel_fiber_[gp].set_state(lambda_f, 1.0);
    remodel_fiber_[gp].Update();

    update_homeostatic_values(params, eleGID);
    if (params_->growth_enabled_)
    {
      const double dt = params.get<double>("delta time");

      remodel_fiber_[gp].integrate_local_evolution_equations_explicit(dt);
    }
  }
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  MixtureConstituent::register_output_data_names(names_and_size);
  names_and_size["mixture_constituent_" + std::to_string(Id()) + "_sig_h"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(Id()) + "_sig"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(Id()) + "_growth_scalar"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(Id()) + "_lambda_r"] = 1;
}

bool MIXTURE::MixtureConstituentRemodelFiberExpl::EvaluateOutputData(
    const std::string& name, CORE::LINALG::SerialDenseMatrix& data) const
{
  if (name == "mixture_constituent_" + std::to_string(Id()) + "_sig_h")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = remodel_fiber_[gp].evaluate_current_homeostatic_fiber_cauchy_stress();
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(Id()) + "_sig")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = remodel_fiber_[gp].evaluate_current_fiber_cauchy_stress();
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(Id()) + "_growth_scalar")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = remodel_fiber_[gp].evaluate_current_growth_scalar();
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(Id()) + "_lambda_r")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = remodel_fiber_[gp].evaluate_current_lambdar();
    }
    return true;
  }
  return MixtureConstituent::EvaluateOutputData(name, data);
}

CORE::LINALG::Matrix<6, 1> MIXTURE::MixtureConstituentRemodelFiberExpl::evaluate_current_p_k2(
    int gp, int eleGID) const
{
  CORE::LINALG::Matrix<6, 1> S_stress(false);
  const double fiber_pk2 = remodel_fiber_[gp].evaluate_current_fiber_p_k2_stress();

  S_stress.Update(fiber_pk2, anisotropy_extension_.get_structural_tensor_stress(gp, 0));

  return S_stress;
}

CORE::LINALG::Matrix<6, 6> MIXTURE::MixtureConstituentRemodelFiberExpl::evaluate_current_cmat(
    const int gp, const int eleGID) const
{
  const double dPK2dlambdafsq =
      remodel_fiber_[gp].evaluate_d_current_fiber_p_k2_stress_d_lambdafsq();

  CORE::LINALG::Matrix<6, 6> cmat(false);
  cmat.MultiplyNT(2.0 * dPK2dlambdafsq, anisotropy_extension_.get_structural_tensor_stress(gp, 0),
      anisotropy_extension_.get_structural_tensor_stress(gp, 0));
  return cmat;
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::Evaluate(const CORE::LINALG::Matrix<3, 3>& F,
    const CORE::LINALG::Matrix<6, 1>& E_strain, Teuchos::ParameterList& params,
    CORE::LINALG::Matrix<6, 1>& S_stress, CORE::LINALG::Matrix<6, 6>& cmat, int gp, int eleGID)
{
  if (params_->inelastic_external_deformation_)
  {
    FOUR_C_THROW(
        "You specified that there is inelastic external deformation in the input file, but this "
        "method is only called if there is none. Probably, you are using a mixture rule without "
        "inelastic growth. You have to set INELASTIC_GROWTH to false or use a different growth "
        "rule.");
  }

  CORE::LINALG::Matrix<3, 3> C = EvaluateC(F);

  const double lambda_f = evaluate_lambdaf(C, gp, eleGID);
  remodel_fiber_[gp].set_state(lambda_f, 1.0);

  S_stress.Update(evaluate_current_p_k2(gp, eleGID));
  cmat.Update(evaluate_current_cmat(gp, eleGID));
}

void MIXTURE::MixtureConstituentRemodelFiberExpl::EvaluateElasticPart(
    const CORE::LINALG::Matrix<3, 3>& FM, const CORE::LINALG::Matrix<3, 3>& iFextin,
    Teuchos::ParameterList& params, CORE::LINALG::Matrix<6, 1>& S_stress,
    CORE::LINALG::Matrix<6, 6>& cmat, int gp, int eleGID)
{
  if (!params_->inelastic_external_deformation_)
  {
    FOUR_C_THROW(
        "You specified that there is no inelastic external deformation in the input file, but this "
        "method is only called if there is one. Probably, you are using a mixture rule with "
        "inelastic growth. You have to set INELASTIC_GROWTH to true or use a different growth "
        "rule.");
  }

  CORE::LINALG::Matrix<3, 3> C = EvaluateC(FM);

  const double lambda_f = evaluate_lambdaf(C, gp, eleGID);
  const double lambda_ext = evaluate_lambda_ext(iFextin, gp, eleGID);
  remodel_fiber_[gp].set_state(lambda_f, lambda_ext);

  S_stress.Update(evaluate_current_p_k2(gp, eleGID));
  cmat.Update(evaluate_current_cmat(gp, eleGID));
}

double MIXTURE::MixtureConstituentRemodelFiberExpl::GetGrowthScalar(int gp) const
{
  return remodel_fiber_[gp].evaluate_current_growth_scalar();
}

double MIXTURE::MixtureConstituentRemodelFiberExpl::evaluate_deposition_stretch(
    const double time) const
{
  if (params_->deposition_stretch_timefunc_num_ == 0)
  {
    return params_->deposition_stretch_;
  }

  return GLOBAL::Problem::Instance()
      ->FunctionById<CORE::UTILS::FunctionOfTime>(params_->deposition_stretch_timefunc_num_ - 1)
      .Evaluate(time);
}
void MIXTURE::MixtureConstituentRemodelFiberExpl::update_homeostatic_values(
    const Teuchos::ParameterList& params, const int eleGID)
{
  // Update deposition stretch / prestretch of fiber depending on time function
  const double time = std::invoke(
      [&]()
      {
        constexpr auto total_time_key = "total time";
        if (!params.isParameter(total_time_key)) return 0.0;

        const double total_time = params.get<double>(total_time_key);
        if (total_time < 0.0) return 0.0;

        return total_time;
      });

  double new_lambda_pre = evaluate_deposition_stretch(time);

  for (auto& fiber : remodel_fiber_)
  {
    fiber.update_deposition_stretch(new_lambda_pre);
  }
}

double MIXTURE::MixtureConstituentRemodelFiberExpl::evaluate_lambdaf(
    const CORE::LINALG::Matrix<3, 3>& C, const int gp, const int eleGID) const
{
  return std::sqrt(C.Dot(anisotropy_extension_.GetStructuralTensor(gp, 0)));
}

double MIXTURE::MixtureConstituentRemodelFiberExpl::evaluate_lambda_ext(
    const CORE::LINALG::Matrix<3, 3>& iFext, const int gp, const int eleGID) const
{
  return 1.0 /
         std::sqrt(EvaluateiCext(iFext).Dot(anisotropy_extension_.GetStructuralTensor(gp, 0)));
}
FOUR_C_NAMESPACE_CLOSE