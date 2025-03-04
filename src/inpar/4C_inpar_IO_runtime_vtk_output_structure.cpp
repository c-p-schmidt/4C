// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_IO_runtime_vtk_output_structure.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Inpar
{
  namespace IORuntimeOutput
  {
    namespace Solid
    {
      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
      {
        using Teuchos::tuple;

        // related sublist
        Core::Utils::SectionSpecs sublist_IO{"IO"};
        Core::Utils::SectionSpecs sublist_IO_VTK{sublist_IO, "RUNTIME VTK OUTPUT"};
        Core::Utils::SectionSpecs sublist_IO_VTK_structure{sublist_IO_VTK, "STRUCTURE"};

        // whether to write output for structure
        Core::Utils::bool_parameter(
            "OUTPUT_STRUCTURE", false, "write structure output", sublist_IO_VTK_structure);

        // whether to write displacement state
        Core::Utils::bool_parameter(
            "DISPLACEMENT", false, "write displacement output", sublist_IO_VTK_structure);

        // whether to write velocity state
        Core::Utils::bool_parameter(
            "VELOCITY", false, "write velocity output", sublist_IO_VTK_structure);

        Core::Utils::bool_parameter(
            "ACCELERATION", false, "write acceleration output", sublist_IO_VTK_structure);

        // whether to write element owner
        Core::Utils::bool_parameter(
            "ELEMENT_OWNER", false, "write element owner", sublist_IO_VTK_structure);

        // whether to write element GIDs
        Core::Utils::bool_parameter(
            "ELEMENT_GID", false, "write 4C internal element GIDs", sublist_IO_VTK_structure);

        // write element ghosting information
        Core::Utils::bool_parameter("ELEMENT_GHOSTING", false,
            "write which processors ghost the elements", sublist_IO_VTK_structure);

        // whether to write node GIDs
        Core::Utils::bool_parameter(
            "NODE_GID", false, "write 4C internal node GIDs", sublist_IO_VTK_structure);

        // write element material IDs
        Core::Utils::bool_parameter("ELEMENT_MAT_ID", false,
            "Output of the material id of each element", sublist_IO_VTK_structure);

        // whether to write stress and / or strain data
        Core::Utils::bool_parameter("STRESS_STRAIN", false,
            "Write element stress and / or strain  data. The type of stress / strain has to be "
            "selected in the --IO input section",
            sublist_IO_VTK_structure);

        // mode to write gauss point data
        Core::Utils::string_to_integral_parameter<Inpar::Solid::GaussPointDataOutputType>(
            "GAUSS_POINT_DATA_OUTPUT_TYPE", "none",
            "Where to write gauss point data. (none, projected to nodes, projected to element "
            "center, raw at gauss points)",
            tuple<std::string>("none", "nodes", "element_center", "gauss_points"),
            tuple<Inpar::Solid::GaussPointDataOutputType>(
                Inpar::Solid::GaussPointDataOutputType::none,
                Inpar::Solid::GaussPointDataOutputType::nodes,
                Inpar::Solid::GaussPointDataOutputType::element_center,
                Inpar::Solid::GaussPointDataOutputType::gauss_points),
            sublist_IO_VTK_structure);

        sublist_IO_VTK_structure.move_into_collection(list);
      }


    }  // namespace Solid
  }  // namespace IORuntimeOutput
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE
