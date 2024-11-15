// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_immersed_problem_fsi_partitioned_immersed.hpp"

#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_fsi_debugwriter.hpp"
#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN


FSI::PartitionedImmersed::PartitionedImmersed(const Epetra_Comm& comm) : Partitioned(comm)
{
  // empty constructor
}


void FSI::PartitionedImmersed::setup()
{
  // call setup of base class
  FSI::Partitioned::setup();
}


void FSI::PartitionedImmersed::setup_coupling(
    const Teuchos::ParameterList& fsidyn, const Epetra_Comm& comm)
{
  if (get_comm().MyPID() == 0)
    std::cout << "\n setup_coupling in FSI::PartitionedImmersed ..." << std::endl;

  // for immersed fsi
  coupsfm_ = Teuchos::null;
  matchingnodes_ = false;

  // enable debugging
  if (fsidyn.get<bool>("DEBUGOUTPUT"))
    debugwriter_ = Teuchos::make_rcp<Utils::DebugWriter>(structure_field()->discretization());
}


void FSI::PartitionedImmersed::extract_previous_interface_solution()
{
  // not necessary in immersed fsi.
  // overrides version in fsi_paritioned with "do nothing".
}

FOUR_C_NAMESPACE_CLOSE
