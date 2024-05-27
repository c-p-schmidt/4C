/*----------------------------------------------------------------------*/
/*! \file
\brief Input parameters for poro problems

\level 3

*------------------------------------------------------------------------------------------------*/

#ifndef FOUR_C_INPAR_PORO_HPP
#define FOUR_C_INPAR_PORO_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace INPAR::PORO
{
  //! poro element implementation type
  enum class PoroType
  {
    undefined,
    pressure_based,
    pressure_velocity_based
  };

}  // namespace INPAR::PORO

FOUR_C_NAMESPACE_CLOSE

#endif