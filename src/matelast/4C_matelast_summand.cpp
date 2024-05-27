/*----------------------------------------------------------------------*/
/*! \file
\brief Implementation of an interface class for materials of the (visco)elasthyper toolbox.

\level 1
*/
/*----------------------------------------------------------------------*/

#include "4C_matelast_summand.hpp"

#include "4C_global_data.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_matelast_aniso_structuraltensor_strategy.hpp"
#include "4C_matelast_anisoactivestress_evolution.hpp"
#include "4C_matelast_coup13apow.hpp"
#include "4C_matelast_coup1pow.hpp"
#include "4C_matelast_coup2pow.hpp"
#include "4C_matelast_coup3pow.hpp"
#include "4C_matelast_coupanisoexpo.hpp"
#include "4C_matelast_coupanisoexposhear.hpp"
#include "4C_matelast_coupanisoexpotwocoup.hpp"
#include "4C_matelast_coupanisoneohooke.hpp"
#include "4C_matelast_coupanisoneohooke_VarProp.hpp"
#include "4C_matelast_coupanisopow.hpp"
#include "4C_matelast_coupblatzko.hpp"
#include "4C_matelast_coupexppol.hpp"
#include "4C_matelast_couplogmixneohooke.hpp"
#include "4C_matelast_couplogneohooke.hpp"
#include "4C_matelast_coupmooneyrivlin.hpp"
#include "4C_matelast_coupneohooke.hpp"
#include "4C_matelast_coupSaintVenantKirchhoff.hpp"
#include "4C_matelast_coupsimopister.hpp"
#include "4C_matelast_couptransverselyisotropic.hpp"
#include "4C_matelast_coupvarga.hpp"
#include "4C_matelast_iso1pow.hpp"
#include "4C_matelast_iso2pow.hpp"
#include "4C_matelast_isoanisoexpo.hpp"
#include "4C_matelast_isoexpopow.hpp"
#include "4C_matelast_isomooneyrivlin.hpp"
#include "4C_matelast_isomuscle_blemker.hpp"
#include "4C_matelast_isoneohooke.hpp"
#include "4C_matelast_isoogden.hpp"
#include "4C_matelast_isotestmaterial.hpp"
#include "4C_matelast_isovarga.hpp"
#include "4C_matelast_isovolaaagasser.hpp"
#include "4C_matelast_isoyeoh.hpp"
#include "4C_matelast_remodelfiber.hpp"
#include "4C_matelast_visco_coupmyocard.hpp"
#include "4C_matelast_visco_fract.hpp"
#include "4C_matelast_visco_generalizedgenmax.hpp"
#include "4C_matelast_visco_genmax.hpp"
#include "4C_matelast_visco_isoratedep.hpp"
#include "4C_matelast_vologden.hpp"
#include "4C_matelast_volpenalty.hpp"
#include "4C_matelast_volpow.hpp"
#include "4C_matelast_volsussmanbathe.hpp"

FOUR_C_NAMESPACE_OPEN

Teuchos::RCP<MAT::ELASTIC::Summand> MAT::ELASTIC::Summand::Factory(int matnum)
{
  // for the sake of safety
  if (GLOBAL::Problem::Instance()->Materials() == Teuchos::null)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");

  // yet another safety check
  if (GLOBAL::Problem::Instance()->Materials()->Num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // retrieve problem instance to read from
  const int probinst = GLOBAL::Problem::Instance()->Materials()->GetReadFromProblem();
  // retrieve validated input line of material ID in question
  Teuchos::RCP<CORE::MAT::PAR::Material> curmat =
      GLOBAL::Problem::Instance(probinst)->Materials()->ById(matnum);

  // construct structural tensor strategy for anisotropic materials
  switch (curmat->Type())
  {
    case CORE::Materials::mes_isoanisoexpo:
    case CORE::Materials::mes_isomuscleblemker:
    case CORE::Materials::mes_coupanisoexpo:
    case CORE::Materials::mes_coupanisoexpoactive:
    case CORE::Materials::mes_coupanisoexpotwocoup:
    case CORE::Materials::mes_coupanisoneohooke:
    case CORE::Materials::mes_coupanisopow:
    case CORE::Materials::mes_coupanisoneohooke_varprop:
    {
      break;
    }
    default:
      break;
  }

  switch (curmat->Type())
  {
    case CORE::Materials::mes_anisoactivestress_evolution:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::AnisoActiveStressEvolution(curmat));
      auto* params =
          dynamic_cast<MAT::ELASTIC::PAR::AnisoActiveStressEvolution*>(curmat->Parameter());
      return Teuchos::rcp(new AnisoActiveStressEvolution(params));
    }
    case CORE::Materials::mes_coupanisoexpoactive:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupAnisoExpoActive(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupAnisoExpoActive*>(curmat->Parameter());
      return Teuchos::rcp(new CoupAnisoExpoActive(params));
    }
    case CORE::Materials::mes_coupanisoexpo:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupAnisoExpo(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupAnisoExpo*>(curmat->Parameter());
      return Teuchos::rcp(new CoupAnisoExpo(params));
    }
    case CORE::Materials::mes_coupanisoexposhear:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupAnisoExpoShear(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupAnisoExpoShear*>(curmat->Parameter());
      return Teuchos::rcp(new CoupAnisoExpoShear(params));
    }
    case CORE::Materials::mes_coupanisoexpotwocoup:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupAnisoExpoTwoCoup(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupAnisoExpoTwoCoup*>(curmat->Parameter());
      return Teuchos::rcp(new CoupAnisoExpoTwoCoup(params));
    }
    case CORE::Materials::mes_coupanisoneohooke:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupAnisoNeoHooke(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupAnisoNeoHooke*>(curmat->Parameter());
      return Teuchos::rcp(new CoupAnisoNeoHooke(params));
    }
    case CORE::Materials::mes_coupanisoneohooke_varprop:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupAnisoNeoHookeVarProp(curmat));
      auto* params =
          dynamic_cast<MAT::ELASTIC::PAR::CoupAnisoNeoHookeVarProp*>(curmat->Parameter());
      return Teuchos::rcp(new CoupAnisoNeoHookeVarProp(params));
    }
    case CORE::Materials::mes_coupanisopow:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupAnisoPow(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupAnisoPow*>(curmat->Parameter());
      return Teuchos::rcp(new CoupAnisoPow(params));
    }
    case CORE::Materials::mes_coupblatzko:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupBlatzKo(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupBlatzKo*>(curmat->Parameter());
      return Teuchos::rcp(new CoupBlatzKo(params));
    }
    case CORE::Materials::mes_coupexppol:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupExpPol(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupExpPol*>(curmat->Parameter());
      return Teuchos::rcp(new CoupExpPol(params));
    }
    case CORE::Materials::mes_couplogneohooke:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupLogNeoHooke(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupLogNeoHooke*>(curmat->Parameter());
      return Teuchos::rcp(new CoupLogNeoHooke(params));
    }
    case CORE::Materials::mes_couplogmixneohooke:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupLogMixNeoHooke(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupLogMixNeoHooke*>(curmat->Parameter());
      return Teuchos::rcp(new CoupLogMixNeoHooke(params));
    }
    case CORE::Materials::mes_coupmooneyrivlin:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupMooneyRivlin(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupMooneyRivlin*>(curmat->Parameter());
      return Teuchos::rcp(new CoupMooneyRivlin(params));
    }
    case CORE::Materials::mes_coupmyocard:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupMyocard(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupMyocard*>(curmat->Parameter());
      return Teuchos::rcp(new CoupMyocard(params));
    }
    case CORE::Materials::mes_coupneohooke:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupNeoHooke(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupNeoHooke*>(curmat->Parameter());
      return Teuchos::rcp(new CoupNeoHooke(params));
    }
    case CORE::Materials::mes_coup1pow:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::Coup1Pow(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::Coup1Pow*>(curmat->Parameter());
      return Teuchos::rcp(new Coup1Pow(params));
    }
    case CORE::Materials::mes_coup2pow:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::Coup2Pow(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::Coup2Pow*>(curmat->Parameter());
      return Teuchos::rcp(new Coup2Pow(params));
    }
    case CORE::Materials::mes_coup3pow:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::Coup3Pow(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::Coup3Pow*>(curmat->Parameter());
      return Teuchos::rcp(new Coup3Pow(params));
    }
    case CORE::Materials::mes_coup13apow:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::Coup13aPow(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::Coup13aPow*>(curmat->Parameter());
      return Teuchos::rcp(new Coup13aPow(params));
    }
    case CORE::Materials::mes_coupsimopister:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupSimoPister(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupSimoPister*>(curmat->Parameter());
      return Teuchos::rcp(new CoupSimoPister(params));
    }
    case CORE::Materials::mes_coupSVK:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupSVK(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupSVK*>(curmat->Parameter());
      return Teuchos::rcp(new CoupSVK(params));
    }
    case CORE::Materials::mes_couptransverselyisotropic:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupTransverselyIsotropic(curmat));
      auto* params =
          dynamic_cast<MAT::ELASTIC::PAR::CoupTransverselyIsotropic*>(curmat->Parameter());
      return Teuchos::rcp(new CoupTransverselyIsotropic(params));
    }
    case CORE::Materials::mes_coupvarga:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::CoupVarga(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::CoupVarga*>(curmat->Parameter());
      return Teuchos::rcp(new CoupVarga(params));
    }
    case CORE::Materials::mes_fract:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::Fract(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::Fract*>(curmat->Parameter());
      return Teuchos::rcp(new Fract(params));
    }
    case CORE::Materials::mes_genmax:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::GenMax(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::GenMax*>(curmat->Parameter());
      return Teuchos::rcp(new GenMax(params));
    }
    case CORE::Materials::mes_generalizedgenmax:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::GeneralizedGenMax(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::GeneralizedGenMax*>(curmat->Parameter());
      return Teuchos::rcp(new GeneralizedGenMax(params));
    }
    case CORE::Materials::mes_isoanisoexpo:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoAnisoExpo(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::IsoAnisoExpo*>(curmat->Parameter());
      return Teuchos::rcp(new IsoAnisoExpo(params));
    }
    case CORE::Materials::mes_isoexpopow:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoExpoPow(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::IsoExpoPow*>(curmat->Parameter());
      return Teuchos::rcp(new IsoExpoPow(params));
    }
    case CORE::Materials::mes_isomooneyrivlin:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoMooneyRivlin(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::IsoMooneyRivlin*>(curmat->Parameter());
      return Teuchos::rcp(new IsoMooneyRivlin(params));
    }
    case CORE::Materials::mes_isomuscleblemker:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoMuscleBlemker(curmat));
      MAT::ELASTIC::PAR::IsoMuscleBlemker* params =
          static_cast<MAT::ELASTIC::PAR::IsoMuscleBlemker*>(curmat->Parameter());
      return Teuchos::rcp(new IsoMuscleBlemker(params));
    }
    case CORE::Materials::mes_isoneohooke:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoNeoHooke(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::IsoNeoHooke*>(curmat->Parameter());
      return Teuchos::rcp(new IsoNeoHooke(params));
    }
    case CORE::Materials::mes_isoogden:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoOgden(curmat));
      auto* params = static_cast<MAT::ELASTIC::PAR::IsoOgden*>(curmat->Parameter());
      return Teuchos::rcp(new IsoOgden(params));
    }
    case CORE::Materials::mes_iso1pow:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::Iso1Pow(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::Iso1Pow*>(curmat->Parameter());
      return Teuchos::rcp(new Iso1Pow(params));
    }
    case CORE::Materials::mes_iso2pow:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::Iso2Pow(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::Iso2Pow*>(curmat->Parameter());
      return Teuchos::rcp(new Iso2Pow(params));
    }
    case CORE::Materials::mes_isoratedep:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoRateDep(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::IsoRateDep*>(curmat->Parameter());
      return Teuchos::rcp(new IsoRateDep(params));
    }
    case CORE::Materials::mes_isotestmaterial:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoTestMaterial(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::IsoTestMaterial*>(curmat->Parameter());
      return Teuchos::rcp(new IsoTestMaterial(params));
    }
    case CORE::Materials::mes_isovarga:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoVarga(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::IsoVarga*>(curmat->Parameter());
      return Teuchos::rcp(new IsoVarga(params));
    }
    case CORE::Materials::mes_isovolaaagasser:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoVolAAAGasser(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::IsoVolAAAGasser*>(curmat->Parameter());
      return Teuchos::rcp(new IsoVolAAAGasser(params));
    }
    case CORE::Materials::mes_isoyeoh:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::IsoYeoh(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::IsoYeoh*>(curmat->Parameter());
      return Teuchos::rcp(new IsoYeoh(params));
    }
    case CORE::Materials::mes_remodelfiber:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::RemodelFiber(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::RemodelFiber*>(curmat->Parameter());
      return Teuchos::rcp(new RemodelFiber(params));
    }
    case CORE::Materials::mes_vologden:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::VolOgden(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::VolOgden*>(curmat->Parameter());
      return Teuchos::rcp(new VolOgden(params));
    }
    case CORE::Materials::mes_volpenalty:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::VolPenalty(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::VolPenalty*>(curmat->Parameter());
      return Teuchos::rcp(new VolPenalty(params));
    }
    case CORE::Materials::mes_volpow:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::VolPow(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::VolPow*>(curmat->Parameter());
      return Teuchos::rcp(new VolPow(params));
    }
    case CORE::Materials::mes_volsussmanbathe:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::VolSussmanBathe(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::VolSussmanBathe*>(curmat->Parameter());
      return Teuchos::rcp(new VolSussmanBathe(params));
    }
    case CORE::Materials::mes_viscobranch:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::ViscoBranch(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::ViscoBranch*>(curmat->Parameter());
      return Teuchos::rcp(new ViscoBranch(params));
    }
    case CORE::Materials::mes_viscopart:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new MAT::ELASTIC::PAR::ViscoPart(curmat));
      auto* params = dynamic_cast<MAT::ELASTIC::PAR::ViscoPart*>(curmat->Parameter());
      return Teuchos::rcp(new ViscoPart(params));
    }
    default:
      FOUR_C_THROW("cannot deal with type %d", curmat->Type());
  }
  return Teuchos::null;
}

void MAT::ELASTIC::Summand::AddShearMod(bool& haveshearmod, double& shearmod) const
{
  FOUR_C_THROW("MAT::ELASTIC::Summand::AddShearMod: Add Shear Modulus not implemented - do so!");
}

int MAT::ELASTIC::Summand::UniqueParObjectId() const { return -1; }

void MAT::ELASTIC::Summand::Pack(CORE::COMM::PackBuffer& data) const { return; }

void MAT::ELASTIC::Summand::Unpack(const std::vector<char>& data) { return; };


// Function which reads in the given fiber value due to the FIBER1 nomenclature
void MAT::ELASTIC::Summand::ReadFiber(INPUT::LineDefinition* linedef, const std::string& specifier,
    CORE::LINALG::Matrix<3, 1>& fiber_vector)
{
  std::vector<double> fiber1;
  linedef->ExtractDoubleVector(specifier, fiber1);
  double f1norm = 0.;
  // normalization
  for (int i = 0; i < 3; ++i)
  {
    f1norm += fiber1[i] * fiber1[i];
  }
  f1norm = sqrt(f1norm);

  // fill final fiber vector
  for (int i = 0; i < 3; ++i) fiber_vector(i) = fiber1[i] / f1norm;
}

// Function which reads in the given fiber value due to the CIR-AXI-RAD nomenclature
void MAT::ELASTIC::Summand::ReadRadAxiCir(
    INPUT::LineDefinition* linedef, CORE::LINALG::Matrix<3, 3>& locsys)
{
  // read local (cylindrical) cosy-directions at current element
  // basis is local cosy with third vec e3 = circumferential dir and e2 = axial dir
  CORE::LINALG::Matrix<3, 1> fiber_rad;
  CORE::LINALG::Matrix<3, 1> fiber_axi;
  CORE::LINALG::Matrix<3, 1> fiber_cir;

  ReadFiber(linedef, "RAD", fiber_rad);
  ReadFiber(linedef, "AXI", fiber_axi);
  ReadFiber(linedef, "CIR", fiber_cir);

  for (int i = 0; i < 3; ++i)
  {
    locsys(i, 0) = fiber_rad(i);
    locsys(i, 1) = fiber_axi(i);
    locsys(i, 2) = fiber_cir(i);
  }
}

void MAT::ELASTIC::Summand::evaluate_first_derivatives_aniso(CORE::LINALG::Matrix<2, 1>& dPI_aniso,
    CORE::LINALG::Matrix<3, 3> const& rcg, int gp, int eleGID)
{
  bool isoprinc, isomod, anisoprinc, anisomod, viscogeneral;
  SpecifyFormulation(isoprinc, isomod, anisoprinc, anisomod, viscogeneral);
  if (anisoprinc or anisomod)
  {
    FOUR_C_THROW(
        "This anisotropic material does not support the first derivative of the free-energy "
        "function with respect to the anisotropic invariants. You need to implement them.");
  }
}

void MAT::ELASTIC::Summand::evaluate_second_derivatives_aniso(
    CORE::LINALG::Matrix<3, 1>& ddPII_aniso, CORE::LINALG::Matrix<3, 3> const& rcg, int gp,
    int eleGID)
{
  bool isoprinc, isomod, anisoprinc, anisomod, viscogeneral;
  SpecifyFormulation(isoprinc, isomod, anisoprinc, anisomod, viscogeneral);
  if (anisoprinc or anisomod)
  {
    FOUR_C_THROW(
        "This anisotropic material does not support the second derivative of the free-energy "
        "function with respect to the anisotropic invariants. You need to implement them.");
  }
}
FOUR_C_NAMESPACE_CLOSE