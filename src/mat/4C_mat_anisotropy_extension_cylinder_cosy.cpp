/*----------------------------------------------------------------------*/
/*! \file

\brief Declaration of a cylinder coordinate system anisotropy extension to be used by anisotropic
materials with @MAT::Anisotropy

\level 3


*/
/*----------------------------------------------------------------------*/

#include "4C_mat_anisotropy_extension_cylinder_cosy.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_anisotropy_coordinate_system_provider.hpp"

FOUR_C_NAMESPACE_OPEN

MAT::CylinderCoordinateSystemAnisotropyExtension::CylinderCoordinateSystemAnisotropyExtension()
    : cosy_location_(CosyLocation::None)
{
}

void MAT::CylinderCoordinateSystemAnisotropyExtension::PackAnisotropy(
    CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::ParObject::AddtoPack(data, static_cast<int>(cosy_location_));
}

void MAT::CylinderCoordinateSystemAnisotropyExtension::UnpackAnisotropy(
    const std::vector<char>& data, std::vector<char>::size_type& position)
{
  cosy_location_ = static_cast<CosyLocation>(CORE::COMM::ParObject::ExtractInt(position, data));
}

void MAT::CylinderCoordinateSystemAnisotropyExtension::on_global_data_initialized()
{
  if (GetAnisotropy()->has_gp_cylinder_coordinate_system())
  {
    cosy_location_ = CosyLocation::GPCosy;
  }
  else if (GetAnisotropy()->has_element_cylinder_coordinate_system())
  {
    cosy_location_ = CosyLocation::ElementCosy;
  }
  else
  {
    cosy_location_ = CosyLocation::None;
  }
}

void MAT::CylinderCoordinateSystemAnisotropyExtension::on_global_element_data_initialized()
{
  // do nothing
}

void MAT::CylinderCoordinateSystemAnisotropyExtension::on_global_gp_data_initialized()
{
  // do nothing
}

const MAT::CylinderCoordinateSystemProvider&
MAT::CylinderCoordinateSystemAnisotropyExtension::get_cylinder_coordinate_system(int gp) const
{
  if (cosy_location_ == CosyLocation::None)
  {
    FOUR_C_THROW("No cylinder coordinate system defined!");
  }

  if (cosy_location_ == CosyLocation::ElementCosy)
  {
    return GetAnisotropy()->get_element_cylinder_coordinate_system();
  }

  return GetAnisotropy()->get_gp_cylinder_coordinate_system(gp);
}

Teuchos::RCP<MAT::CoordinateSystemProvider>
MAT::CylinderCoordinateSystemAnisotropyExtension::get_coordinate_system_provider(int gp) const
{
  auto cosy = Teuchos::rcp(new CoordinateSystemHolder());

  if (cosy_location_ != CosyLocation::None)
    cosy->set_cylinder_coordinate_system_provider(
        Teuchos::rcpFromRef(get_cylinder_coordinate_system(gp)));

  return cosy;
}
FOUR_C_NAMESPACE_CLOSE