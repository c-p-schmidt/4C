/*----------------------------------------------------------------------*/
/*! \file
\brief permeable fluid

\level 2

*/
/*----------------------------------------------------------------------*/


#include "4C_mat_permeablefluid.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::PermeableFluid::PermeableFluid(Teuchos::RCP<CORE::MAT::PAR::Material> matdata)
    : Parameter(matdata),
      type_(matdata->Get<std::string>("TYPE")),
      viscosity_(matdata->Get<double>("DYNVISCOSITY")),
      density_(matdata->Get<double>("DENSITY")),
      permeability_(matdata->Get<double>("PERMEABILITY"))
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<CORE::MAT::Material> MAT::PAR::PermeableFluid::CreateMaterial()
{
  return Teuchos::rcp(new MAT::PermeableFluid(this));
}


MAT::PermeableFluidType MAT::PermeableFluidType::instance_;


CORE::COMM::ParObject* MAT::PermeableFluidType::Create(const std::vector<char>& data)
{
  MAT::PermeableFluid* permeable_fluid = new MAT::PermeableFluid();
  permeable_fluid->Unpack(data);
  return permeable_fluid;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PermeableFluid::PermeableFluid() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PermeableFluid::PermeableFluid(MAT::PAR::PermeableFluid* params) : params_(params) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PermeableFluid::Pack(CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data, matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PermeableFluid::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  CORE::COMM::ExtractAndAssertId(position, data, UniqueParObjectId());

  // matid
  int matid;
  ExtractfromPack(position, data, matid);
  params_ = nullptr;
  if (GLOBAL::Problem::Instance()->Materials() != Teuchos::null)
    if (GLOBAL::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = GLOBAL::Problem::Instance()->Materials()->GetReadFromProblem();
      CORE::MAT::PAR::Parameter* mat =
          GLOBAL::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::PermeableFluid*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());
    }

  if (position != data.size())
    FOUR_C_THROW("Mismatch in size of data %d <-> %d", data.size(), position);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double MAT::PermeableFluid::compute_reaction_coeff() const
{
  // check for zero or negative viscosity
  if (Viscosity() < 1e-15) FOUR_C_THROW("zero or negative viscosity");

  // check for zero or negative permeability
  if (Permeability() < 1e-15) FOUR_C_THROW("zero or negative permeability");

  // viscosity divided by permeability
  double reacoeff = Viscosity() / Permeability();

  return reacoeff;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double MAT::PermeableFluid::SetViscosity() const
{
  // set zero viscosity and only modify it for Darcy-Stokes problems
  double viscosity = 0.0;
  if (Type() == "Darcy-Stokes") viscosity = Viscosity();

  return viscosity;
}

FOUR_C_NAMESPACE_CLOSE