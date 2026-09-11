// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_visualization_writer_vtu_collective.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_io_visualization_data.hpp"

#include <cmath>
#include <filesystem>
#include <limits>

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
Core::IO::VisualizationWriterVtuCollective::VisualizationWriterVtuCollective(
    const Core::IO::VisualizationParameters& parameters, MPI_Comm comm,
    std::string visualization_data_name)
    : VisualizationWriterBase(parameters, comm, std::move(visualization_data_name)),
      vtu_writer_(Core::Communication::my_mpi_rank(comm), Core::Communication::num_mpi_ranks(comm),
          std::pow(10, Core::IO::get_total_digits_to_reserve_in_time_step(parameters)),
          parameters.directory_name_, (parameters.file_name_prefix_ + "-vtk-files"),
          visualization_data_name_, parameters.restart_from_name_, parameters.restart_time_,
          parameters.data_format_ == OutputDataFormat::binary, parameters.compression_level_)
{
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::initialize_time_step(
    const double visualization_time, const int visualization_step)
{
  vtu_writer_.reset_time_and_time_step(visualization_time, visualization_step);

  vtu_writer_.initialize_current_time_step_output_file_name();

  // reset the buffers for the current time step
  piece_buffer_.str("");
  piece_buffer_.clear();
  header_buffer_.str("");
  header_buffer_.clear();
  dummy_master_.str("");
  dummy_master_.clear();
  piece_opened_ = false;

  if (Core::Communication::my_mpi_rank(comm_) == 0)
  {
    // write the VTU file header (including the <FieldData> section) on proc 0 only
    vtu_writer_.write_vtk_file_header(header_buffer_);

    vtu_writer_.append_master_file_and_time_to_collection_file_mid_section_content(
        std::filesystem::path(vtu_writer_.output_file_name_shared()).filename().string());
  }
  else
  {
    // set the writer into the initial phase without contributing bytes to the file
    vtu_writer_.write_vtk_file_header(dummy_master_);
  }
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::write_field_data_to_disk(
    const std::map<std::string, visualization_vector_type_variant>& field_data_map)
{
  if (Core::Communication::my_mpi_rank(comm_) == 0)
    vtu_writer_.write_vtk_field_data_and_or_time_and_or_cycle(header_buffer_, field_data_map);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::write_geometry_to_disk(
    const std::vector<double>& point_coordinates,
    const std::vector<Core::IO::index_type>& point_cell_connectivity,
    const std::vector<Core::IO::index_type>& cell_offset, const std::vector<uint8_t>& cell_types,
    const std::vector<Core::IO::index_type>& face_connectivity,
    const std::vector<Core::IO::index_type>& face_offset)
{
  const bool has_points = !point_coordinates.empty();
  const bool has_cells = !cell_types.empty();

  piece_opened_ = has_points || has_cells;

  if (not piece_opened_) return;

  vtu_writer_.write_geometry_unstructured_grid(piece_buffer_, dummy_master_, point_coordinates,
      point_cell_connectivity, cell_offset, cell_types, face_connectivity, face_offset);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::write_point_data_vector_to_disk(
    const visualization_vector_type_variant& data, unsigned int num_components_per_point,
    const std::string& name)
{
  if (not piece_opened_) return;

  vtu_writer_.write_point_data_vector(
      piece_buffer_, dummy_master_, data, num_components_per_point, name);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::write_cell_data_vector_to_disk(
    const visualization_vector_type_variant& data, unsigned int num_components_per_point,
    const std::string& name)
{
  if (not piece_opened_) return;

  vtu_writer_.write_cell_data_vector(
      piece_buffer_, dummy_master_, data, num_components_per_point, name);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::finalize_time_step()
{
  // close the current piece (or write a minimal empty piece if this rank has no content)
  if (piece_opened_)
    vtu_writer_.write_vtk_piece_footer(piece_buffer_, dummy_master_);
  else
    piece_buffer_ << "<Piece NumberOfPoints=\"0\" NumberOfCells=\"0\"/>\n";

  // serialize the VTU file footer into a separate buffer (only the last rank writes it)
  std::ostringstream file_footer_buffer;
  vtu_writer_.write_vtk_file_footer(file_footer_buffer);
  const std::string file_footer = file_footer_buffer.str();

  const int myrank = Core::Communication::my_mpi_rank(comm_);
  const int numproc = Core::Communication::num_mpi_ranks(comm_);

  // determine the size of the header written by proc 0
  uint64_t header_size = header_buffer_.str().size();
  MPI_Bcast(&header_size, 1, MPI_UINT64_T, 0, comm_);

  // determine the size of this processor's piece and its offset in the file
  const std::string piece = piece_buffer_.str();
  uint64_t piece_size = piece.size();
  uint64_t piece_offset = 0;
  MPI_Exscan(&piece_size, &piece_offset, 1, MPI_UINT64_T, MPI_SUM, comm_);
  if (myrank == 0) piece_offset = 0;

  // total size of all pieces
  uint64_t total_piece_size = 0;
  MPI_Allreduce(&piece_size, &total_piece_size, 1, MPI_UINT64_T, MPI_SUM, comm_);

  // open the shared file for collective writing
  MPI_File file_handle = MPI_FILE_NULL;
  const int ierr = MPI_File_open(comm_, vtu_writer_.output_file_name_shared().c_str(),
      MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file_handle);

  // Check every MPI-IO result and coordinate the error across all ranks before throwing so that no
  // rank leaves a collective call while its peers are still waiting inside it.
  const auto throw_if_mpi_io_failed = [this](const int ierr, const std::string& message)
  {
    int max_ierr = ierr;
    MPI_Allreduce(MPI_IN_PLACE, &max_ierr, 1, MPI_INT, MPI_MAX, comm_);
    if (max_ierr != MPI_SUCCESS) FOUR_C_THROW("{}", message);
  };

  throw_if_mpi_io_failed(ierr, "MPI_File_open failed for the shared VTU file.");

  // truncate a possibly existing file from a previous run
  throw_if_mpi_io_failed(
      MPI_File_set_size(file_handle, 0), "Truncating the shared VTU file failed.");
  throw_if_mpi_io_failed(MPI_Barrier(comm_), "Synchronizing the ranks before writing failed.");

  if (header_size > std::numeric_limits<int>::max())
    FOUR_C_THROW("VTU header exceeds the maximum supported size.");

  // proc 0 writes the header (with the field data) at the beginning of the file
  int write_err = MPI_SUCCESS;
  if (myrank == 0)
  {
    write_err = MPI_File_write_at(file_handle, 0, header_buffer_.str().data(),
        static_cast<int>(header_size), MPI_BYTE, MPI_STATUS_IGNORE);
  }
  throw_if_mpi_io_failed(write_err, "Writing the VTU file header failed.");

  // determine the maximum piece size across all ranks so every rank takes the same branch
  uint64_t max_piece_size = 0;
  MPI_Allreduce(&piece_size, &max_piece_size, 1, MPI_UINT64_T, MPI_MAX, comm_);

  if (max_piece_size > std::numeric_limits<int>::max())
    FOUR_C_THROW("VTU piece exceeds the maximum supported size.");

  // all processors write their piece content at the correct offset
  const MPI_Offset write_offset = header_size + piece_offset;
  throw_if_mpi_io_failed(MPI_File_write_at_all(file_handle, write_offset, piece.data(),
                             static_cast<int>(piece_size), MPI_BYTE, MPI_STATUS_IGNORE),
      "Writing the collective VTU piece to the shared file failed.");

  if (file_footer.size() > std::numeric_limits<int>::max())
    FOUR_C_THROW("VTU footer exceeds the maximum supported size.");

  // the last processor appends the file footer
  write_err = MPI_SUCCESS;
  if (myrank == numproc - 1)
  {
    const MPI_Offset footer_offset = header_size + total_piece_size;
    write_err = MPI_File_write_at(file_handle, footer_offset, file_footer.data(),
        static_cast<int>(file_footer.size()), MPI_BYTE, MPI_STATUS_IGNORE);
  }
  throw_if_mpi_io_failed(write_err, "Writing the VTU file footer failed.");

  throw_if_mpi_io_failed(MPI_File_sync(file_handle), "Synchronizing the shared VTU file failed.");
  throw_if_mpi_io_failed(MPI_File_close(&file_handle), "Closing the shared VTU file failed.");

  // Write a collection file summarizing all previously written files
  vtu_writer_.write_vtk_collection_file_for_all_written_master_files(
      parameters_.file_name_prefix_ + "-" + visualization_data_name_);
}

FOUR_C_NAMESPACE_CLOSE