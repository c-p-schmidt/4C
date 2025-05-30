// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_FACTORY_HPP
#define FOUR_C_GEOMETRY_PAIR_FACTORY_HPP


#include "4C_config.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// Forward declarations.
namespace Core::Elements
{
  class Element;
}
namespace GeometryPair
{
  class GeometryPairBase;
  class GeometryEvaluationDataBase;
}  // namespace GeometryPair


namespace GeometryPair
{
  /**
   * \brief Create the correct geometry pair for line to volume coupling.
   * @return RCP to created geometry pair.
   */
  template <typename ScalarType, typename Line, typename Volume>
  std::shared_ptr<GeometryPairBase> geometry_pair_line_to_volume_factory(
      const Core::Elements::Element* element1, const Core::Elements::Element* element2,
      const std::shared_ptr<GeometryEvaluationDataBase>& geometry_evaluation_data_ptr);

  /**
   * \brief Create the correct geometry pair for line to surface coupling.
   * @return RCP to created geometry pair.
   */
  template <typename ScalarType, typename Line, typename Surface>
  std::shared_ptr<GeometryPairBase> geometry_pair_line_to_surface_factory(
      const Core::Elements::Element* element1, const Core::Elements::Element* element2,
      const std::shared_ptr<GeometryEvaluationDataBase>& geometry_evaluation_data_ptr);

  /**
   * \brief Create the correct geometry pair for line to surface coupling with FAD scalar types.
   *
   * The default geometry_pair_line_to_surface_factory would be sufficient for this, however, for
   * performance reasons it is better use the wrapped pairs, which are created in this function.
   *
   * @return RCP to created geometry pair.
   */
  template <typename ScalarType, typename Line, typename Surface>
  std::shared_ptr<GeometryPairBase> geometry_pair_line_to_surface_factory_fad(
      const Core::Elements::Element* element1, const Core::Elements::Element* element2,
      const std::shared_ptr<GeometryEvaluationDataBase>& geometry_evaluation_data_ptr);
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif
