// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_VTU_WRITER_HPP
#define FOUR_C_IO_VTU_WRITER_HPP


#include "4C_config.hpp"

#include "4C_io_vtk_writer_base.hpp"

#include <fstream>
#include <map>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*
 \brief class for VTU output generation

*/
class VtuWriter : public VtkWriterBase
{
 public:
  //! constructor
  VtuWriter(unsigned int myrank, unsigned int num_processors,
      unsigned int max_number_timesteps_to_be_written,
      const std::string& path_existing_working_directory,
      const std::string& name_new_vtk_subdirectory, const std::string& geometry_name,
      const std::string& restart_name, double restart_time, bool write_binary_output,
      LibB64::CompressionLevel compression_level);

  //! open the processor-specific vtk file streams for this processor and master file and set up
  //! file names for the new geometry/time step
  void initialize_vtk_file_streams_for_new_geometry_and_or_time_step();

  //! write the prologue of the vtk files
  void write_vtk_headers();

  //! write the geometry defining this unstructured grid
  void write_geometry_unstructured_grid(const std::vector<double>& point_coordinates,
      const std::vector<Core::IO::index_type>& point_cell_connectivity,
      const std::vector<Core::IO::index_type>& cell_offset, const std::vector<uint8_t>& cell_types,
      const std::vector<Core::IO::index_type>& face_connectivity,
      const std::vector<Core::IO::index_type>& face_offset);


  //! write a data vector with num_component values of type T per point
  void write_point_data_vector(const Core::IO::visualization_vector_type_variant& data,
      unsigned int num_components_per_point, const std::string& name);

  //! write a data vector with num_component values of type T per cell
  void write_cell_data_vector(const Core::IO::visualization_vector_type_variant& data,
      unsigned int num_components_per_cell, const std::string& name);

  //! write field data array to the VTK file on this processor
  void write_vtk_field_data_and_or_time_and_or_cycle(
      const std::map<std::string, Core::IO::visualization_vector_type_variant>& field_data_map);

  //! write field data array to the VTK file on this processor [for restart information]
  void write_vtk_time_and_or_cycle();

  //! write the epilogue of the vtk files
  void write_vtk_footers();


 protected:
  //! write a data vector as DataArray to corresponding vtk files
  // Todo template <typename T>
  void write_data_array(const Core::IO::visualization_vector_type_variant& data,
      const int num_components, const std::string& name);

  //! Return the opening xml tag for this writer type
  const std::string& writer_opening_tag() const override;

  //! Return the parallel opening xml tag for this writer type
  const std::string& writer_p_opening_tag() const override;

  //! Return a vector of parallel piece tags for each file
  const std::vector<std::string>& writer_p_piece_tags() const override;

  //! Return the parallel file suffix including the dot for this file type
  const std::string& writer_p_suffix() const override;

  //! Return the string of this writer type
  const std::string& writer_string() const override;

  //! Return the file suffix including the dot for this file type
  const std::string& writer_suffix() const override;

 private:
  //! write prologue of the VTK master file (handled by proc 0)
  void write_vtk_header_master_file(const std::string& byteorder);

  //! write prologue of the VTK file on this processor
  void write_vtk_header_this_processor(const std::string& byteorder);

  //! write field data array to the VTK file on this processor
  template <typename T>
  void write_field_data_array(const std::string& name, const std::vector<T>& field_data);

  //! write the required information about the DataArray to master file
  // Todo template <typename T>
  void write_data_array_master_file(
      const int num_components, const std::string& name, const std::string& data_type_name);

  //! write the data array to this processor's file
  template <typename T>
  void write_data_array_this_processor(
      const std::vector<T>& data, const int num_components, const std::string& name);

  //! write epilogue of of the VTK master file (handled by proc 0)
  void write_vtk_footer_master_file();

  //! write epilogue of the VTK file on this processor
  void write_vtk_footer_this_processor();

  //! initialize the individual vtk file stream on each processor
  void initialize_vtk_file_stream_this_processor();

  //! initialize the vtk 'master file' stream (handled by proc 0)
  void initialize_vtk_master_file_stream();

  //! Output stream for current processor-specific file
  std::ofstream currentout_;

  //! Output stream for current master file (only proc 0)
  std::ofstream currentmasterout_;
};

FOUR_C_NAMESPACE_CLOSE

#endif
