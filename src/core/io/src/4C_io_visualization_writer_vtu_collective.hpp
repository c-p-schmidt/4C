// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_VISUALIZATION_WRITER_VTU_COLLECTIVE_HPP
#define FOUR_C_IO_VISUALIZATION_WRITER_VTU_COLLECTIVE_HPP

#include "4C_config.hpp"

#include "4C_io_visualization_writer_base.hpp"
#include "4C_io_vtu_writer.hpp"

#include <mpi.h>

#include <map>
#include <sstream>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  /**
   * @brief Writer that writes one shared VTU file per time step for all ranks via MPI-IO
   *
   * Each rank serializes its own <Piece> into a buffer; the pieces are concatenated into a single
   * VTU file by means of collective MPI file I/O (MPI_File_write_at_all).
   */
  class VisualizationWriterVtuCollective : public VisualizationWriterBase
  {
   public:
    /**
     * @brief Default constructor (derived)
     */
    VisualizationWriterVtuCollective(const Core::IO::VisualizationParameters& parameters,
        MPI_Comm comm, std::string visualization_data_name);

    /**
     * @brief Default destructor (derived)
     */
    ~VisualizationWriterVtuCollective() override = default;

    /**
     * @brief Initialize the current time step (derived)
     */
    void initialize_time_step(
        const double visualization_time, const int visualization_step) override;

    /**
     * @brief Write all fields contained in the field data map to disk (derived)
     */
    void write_field_data_to_disk(
        const std::map<std::string, visualization_vector_type_variant>& field_data_map) override;

    /**
     * @brief Write the full geometry, i.e., points, cells, faces and the respective connectivity to
     * disk (derived)
     */
    void write_geometry_to_disk(const std::vector<double>& point_coordinates,
        const std::vector<Core::IO::index_type>& point_cell_connectivity,
        const std::vector<Core::IO::index_type>& cell_offset,
        const std::vector<uint8_t>& cell_types,
        const std::vector<Core::IO::index_type>& face_connectivity,
        const std::vector<Core::IO::index_type>& face_offset) override;

    /**
     * @brief Write a single point data vector to disk (derived)
     */
    void write_point_data_vector_to_disk(const visualization_vector_type_variant& data,
        unsigned int num_components_per_point, const std::string& name) override;

    /**
     * @brief Write a single cell data vector to disk (derived)
     */
    void write_cell_data_vector_to_disk(const visualization_vector_type_variant& data,
        unsigned int num_components_per_point, const std::string& name) override;

    /**
     * @brief Finalize the write operations for the current time step (derived)
     */
    void finalize_time_step() override;

   private:
    //! VtuWriter used for the VTU format serialization
    VtuWriter vtu_writer_;

    //! buffer for the <Piece> content of this processor
    std::ostringstream piece_buffer_;

    //! buffer for the VTU file header + field data (only proc 0)
    std::ostringstream header_buffer_;

    //! dummy stream that swallows the parallel master file content (not needed for a single file)
    std::ostringstream dummy_master_;

    //! flag indicating whether this processor opened a non-empty <Piece>
    bool piece_opened_ = false;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif