/*----------------------------------------------------------------------*/
/*! \file

  \brief main routine of the Ensight filter

  \level 1

*/



#include "4C_post_processor_single_field_writers.hpp"

#include "4C_discretization_condition_utils.hpp"
#include "4C_lib_discret.hpp"
#include "4C_post_common.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureFilter::WriteAllResults(PostField* field)
{
  writer_->WriteResult("displacement", "displacement", dofbased, field->problem()->num_dim());
  writer_->WriteResult(
      "prolongated_gauss_2PK_stresses_xyz", "prolongated_gauss_2PK_stresses_xyz", nodebased, 6);
  writer_->WriteResult(
      "prolongated_gauss_GL_strains_xyz", "prolongated_gauss_GL_strains_xyz", nodebased, 6);
  if (field->problem()->struct_vel_acc() == "yes")
  {
    writer_->WriteResult("velocity", "velocity", dofbased, field->problem()->num_dim());
    writer_->WriteResult("acceleration", "acceleration", dofbased, field->problem()->num_dim());
  }
  if (field->problem()->struct_mat_disp() == "yes")
    writer_->WriteResult(
        "material_displacement", "material_displacement", dofbased, field->problem()->num_dim());

  // contact and meshtying results
  writer_->WriteResult("activeset", "activeset", nodebased, 1);
  writer_->WriteResult("contactowner", "contactowner", nodebased, 1);
  writer_->WriteResult(
      "norcontactstress", "norcontactstress", dofbased, field->problem()->num_dim());
  writer_->WriteResult(
      "tancontactstress", "tancontactstress", dofbased, field->problem()->num_dim());
  writer_->WriteResult(
      "interfacetraction", "interfacetraction", dofbased, field->problem()->num_dim());
  writer_->WriteResult("slaveforces", "slaveforces", dofbased, field->problem()->num_dim());
  writer_->WriteResult("masterforces", "masterforces", dofbased, field->problem()->num_dim());
  writer_->WriteResult("norslaveforce", "norslaveforce", dofbased, field->problem()->num_dim());
  writer_->WriteResult("tanslaveforce", "tanslaveforce", dofbased, field->problem()->num_dim());
  writer_->WriteResult("normasterforce", "normasterforce", dofbased, field->problem()->num_dim());
  writer_->WriteResult("tanmasterforce", "tanmasterforce", dofbased, field->problem()->num_dim());
  writer_->WriteResult("wear", "wear", dofbased, field->problem()->num_dim());
  writer_->WriteResult("norslaveforcelm", "norslaveforcelm", dofbased, field->problem()->num_dim());
  writer_->WriteResult("norslaveforceg", "norslaveforceg", dofbased, field->problem()->num_dim());
  writer_->WriteResult(
      "normasterforcelm", "normasterforcelm", dofbased, field->problem()->num_dim());
  writer_->WriteResult("normasterforceg", "normasterforceg", dofbased, field->problem()->num_dim());
  writer_->WriteResult(
      "poronopen_lambda", "poronopen_lambda", dofbased, field->problem()->num_dim());

  // spring dashpot
  writer_->WriteResult("gap", "gap", nodebased, 1);
  writer_->WriteResult("curnormals", "curnormals", nodebased, 3);
  writer_->WriteResult("springstress", "springstress", nodebased, 3);

  // error norms
  writer_->WriteResult("L2_norm", "L2_norm", elementbased, 1);
  writer_->WriteResult("H1_norm", "H1_norm", elementbased, 1);
  writer_->WriteResult("Energy_norm", "Energy_norm", elementbased, 1);

  // one-dimensional artery
  writer_->WriteResult("one_d_artery_pressure", "pressure", dofbased, 1);
  writer_->WriteResult("one_d_artery_flow", "flow", nodebased, 1);
  writer_->WriteResult("one_d_artery_area", "area", nodebased, 1);

  // reduced dimensional airway
  writer_->WriteResult("pnp", "pressure", dofbased, 1);
  writer_->WriteResult("p_nonlin", "pressure_error", dofbased, 1);
  writer_->WriteResult("NodeIDs", "NodeIDs", dofbased, 1);
  writer_->WriteResult("radii", "radii", dofbased, 1);
  //  writer_->WriteResult("juncVolMix", "juncVolMix", dofbased, 1);
  //  writer_->WriteResult("forwardVolume","forwardVolume", dofbased, 1);
  writer_->WriteResult("scatraO2np", "scatraO2", dofbased, 1);
  writer_->WriteResult("PO2", "PO2", dofbased, 1);
  writer_->WriteResult("dVO2", "dVO2", dofbased, 1);
  writer_->WriteResult("AcinarPO2", "AcinarPO2", dofbased, 1);
  //  writer_->WriteResult("e1scatranp","e1scatranp",elementbased, 1);
  //  writer_->WriteResult("e2scatranp","e2scatranp",elementbased, 1);
  writer_->WriteResult("juncVolMix", "juncVolMix", dofbased, 1);

  writer_->WriteResult("acini_vnp", "acini_volume", elementbased, 1);
  writer_->WriteResult("acini_volumetric_strain", "acini_volume_strain", elementbased, 1);
  writer_->WriteResult("acini_max_strain_location", "acini_max_strain_location", elementbased, 1);
  writer_->WriteResult("acin_bc", "acini_bc", elementbased, 1);
  writer_->WriteResult("acini_v0", "acini_volume0", elementbased, 1);

  writer_->WriteResult("qin_np", "flow_in", elementbased, 1);
  writer_->WriteResult("qout_np", "flow_out", elementbased, 1);
  writer_->WriteResult("x_np", "x_np", elementbased, 1);
  writer_->WriteResult("open", "open", elementbased, 1);
  writer_->WriteResult("p_extnp", "p_extnp", elementbased, 1);
  writer_->WriteResult("p_extn", "p_extn", elementbased, 1);
  writer_->WriteResult("airway_acinus_dep", "airway_acinus_dep", elementbased, 1);
  writer_->WriteResult("generations", "generations", elementbased, 1);
  writer_->WriteResult("elemVolumenp", "airway_volume", elementbased, 1);
  writer_->WriteResult("elemVolume0", "airway_volume0", elementbased, 1);
  writer_->WriteResult("elemRadius_current", "radius_current", elementbased, 1);


  // additional forces due to lung fsi (volume constraint)
  writer_->WriteResult("Add_Forces", "Add_Forces", dofbased, field->problem()->num_dim());

  // Lagrange multiplier at the interface in monolithic fsi
  writer_->WriteResult("fsilambda", "fsilambda", dofbased, field->problem()->num_dim());

  // Lagrange multiplier at the interface in monolithic fsi
  writer_->WriteResult("fpilambda_ps", "fpilambda_ps", dofbased, field->problem()->num_dim());
  writer_->WriteResult("fpilambda_pf", "fpilambda_pf", dofbased, field->problem()->num_dim());

  // additional output for biofilm problems
  writer_->WriteResult(
      "str_growth_displ", "str_growth_displ", dofbased, field->problem()->num_dim());

  // additional output for poro problems
  writer_->WriteResult("porosity_p1", "porosity_p1", dofbased, 1);
  writer_->WriteResult(
      "poronopencond_lambda", "poronopencond_lambda", dofbased, field->problem()->num_dim());

  // additional output for tsi (non-matching grid)
  writer_->WriteResult("struct_temperature", "struct_temperature", nodebased, 1);

  // monolithic scalar-structure interaction involving scatra-scatra interface coupling outputs
  // nodal Cauchy stresses instead of Gauss-point ones
  writer_->WriteResult("nodal_stresses_xyz", "nodal_stresses_xyz", nodebased, 6);

  // elastohydrodynamic lubrication
  writer_->WriteResult("fluid_force", "fluid_force", dofbased, field->problem()->num_dim());
  writer_->WriteResult("normal_contact", "normal_contact", dofbased, field->problem()->num_dim());
  writer_->WriteResult(
      "tangential_contact", "tangential_contact", dofbased, field->problem()->num_dim());
  writer_->WriteResult("active", "active", nodebased, 1);
  writer_->WriteResult("slip", "slip", nodebased, 1);

  if (stresstype_ != "none")
  {
    // although appearing here twice, only one function call to PostStress
    // is really postprocessing Gauss point stresses, since only _either_
    // Cauchy _or_ 2nd Piola-Kirchhoff stresses are written during simulation!
    PostStress("gauss_cauchy_stresses_xyz", stresstype_);
    PostStress("gauss_2PK_stresses_xyz", stresstype_);
    // in case of coupled problem (e.g. TSI) write the stresses arising due to
    // coupling to another field
    PostStress("gauss_cauchy_coupling_stresses_xyz", stresstype_);
    PostStress("gauss_2PK_coupling_stresses_xyz", stresstype_);
  }
  if (straintype_ != "none")
  {
    // although appearing here twice, only one function call to PostStress
    // is really postprocessing Gauss point strains, since only _either_
    // Green-Lagrange _or_ Euler-Almansi strains are written during simulation!
    PostStress("gauss_GL_strains_xyz", straintype_);
    PostStress("gauss_EA_strains_xyz", straintype_);
    PostStress("gauss_LOG_strains_xyz", straintype_);
    // the same for plastic strains
    PostStress("gauss_pl_GL_strains_xyz", straintype_);
    PostStress("gauss_pl_EA_strains_xyz", straintype_);
  }
  if (optquantitytype_ != "none")
  {
    PostStress("gauss_membrane_thickness", optquantitytype_);
  }

  // write structural rotation tensor R
  if (field->problem()->struct_rot() == "yes") PostStress("rotation", "cxyz");

  // Write element and node owners
  WriteElementResults(field);
  //  WriteNodeResults(field); // Uncomment to visualize owner proc on a nodal basis

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureFilter::write_all_results_one_time_step(
    PostResult& result, bool firststep, bool laststep)
{
  writer_->write_result_one_time_step(result, "displacement", "displacement", dofbased,
      result.field()->problem()->num_dim(), firststep, laststep);
  WriteNodeResults(result.field());
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MortarFilter::WriteAllResults(PostField* field)
{
  writer_->WriteResult("displacement", "displacement", dofbased, field->problem()->num_dim());

  writer_->WriteResult(
      "norcontactstress", "norcontactstress", dofbased, field->problem()->num_dim());
  writer_->WriteResult(
      "tancontactstress", "tancontactstress", dofbased, field->problem()->num_dim());

  writer_->WriteResult(
      "interfacetraction", "interfacetraction", dofbased, field->problem()->num_dim());

  writer_->WriteResult("slaveforces", "slaveforces", dofbased, field->problem()->num_dim());
  writer_->WriteResult("masterforces", "masterforces", dofbased, field->problem()->num_dim());

  writer_->WriteResult("norslaveforce", "norslaveforce", dofbased, field->problem()->num_dim());
  writer_->WriteResult("tanslaveforce", "tanslaveforce", dofbased, field->problem()->num_dim());

  writer_->WriteResult("normasterforce", "normasterforce", dofbased, field->problem()->num_dim());
  writer_->WriteResult("tanmasterforce", "tanmasterforce", dofbased, field->problem()->num_dim());

  WriteNodeResults(field);
  WriteElementResults(field);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FluidFilter::WriteAllResults(PostField* field)
{
  writer_->WriteResult("averaged_pressure", "averaged_pressure", dofbased, 1);
  writer_->WriteResult(
      "averaged_velnp", "averaged_velocity", dofbased, field->problem()->num_dim());
  writer_->WriteResult(
      "averaged_scanp", "averaged_scalar_field", dofbased, 1, field->problem()->num_dim());
  writer_->WriteResult("filteredvel", "filteredvel", dofbased, field->problem()->num_dim());
  writer_->WriteResult("fsvelaf", "fsvel", dofbased, field->problem()->num_dim());
  writer_->WriteResult("velnp", "velocity", dofbased, field->problem()->num_dim());
  writer_->WriteResult("pressure", "pressure", dofbased, 1);
  writer_->WriteResult("scalar_field", "scalar_field", dofbased, 1);
  writer_->WriteResult("residual", "residual", dofbased, field->problem()->num_dim());
  writer_->WriteResult("dispnp", "ale_displacement", dofbased, field->problem()->num_dim());
  writer_->WriteResult("idispnfull", "ale_idisp", dofbased, field->problem()->num_dim());
  writer_->WriteResult("traction", "traction", dofbased, field->problem()->num_dim());
  writer_->WriteResult("wss", "wss", dofbased, field->problem()->num_dim());
  writer_->WriteResult("wss_mean", "wss_mean", dofbased, field->problem()->num_dim());

  // for xwall
  writer_->WriteResult("xwall_enrvelnp", "xwall_enrvelnp", dofbased, field->problem()->num_dim());
  writer_->WriteResult("xwall_tauw", "xwall_tauw", nodebased, 1);
  //  writer_->WriteResult("radii", "radii", nodebased, 1);
  writer_->WriteResult("par_vel", "par_vel", dofbased, field->problem()->num_dim());

  // additional forces due to lung fsi (volume constraint)
  writer_->WriteResult("Add_Forces", "Add_Forces", dofbased, field->problem()->num_dim());

  // additional output for poro problems
  writer_->WriteResult("convel", "convective_velocity", dofbased, field->problem()->num_dim());
  writer_->WriteResult("gridv", "grid_velocity", dofbased, field->problem()->num_dim());

  // additional fields due to adjoint equations
  writer_->WriteResult("adjoint_velnp", "adjoint_velocity", dofbased, field->problem()->num_dim());
  writer_->WriteResult("adjoint_pressure", "adjoint_pressure", dofbased, 1);

  // additional fields for meshfree problems
  writer_->WriteResult(
      "velatmeshfreenodes", "velocity_atnodes", dofbased, field->problem()->num_dim());
  writer_->WriteResult("pressureatmeshfreenodes", "pressure_atnodes", dofbased, 1);

  // Lagrange multiplier at the interface in monolithic fsi
  writer_->WriteResult("fsilambda", "fsilambda", dofbased, field->problem()->num_dim());

  // additional output for biofilm problems
  writer_->WriteResult(
      "fld_growth_displ", "fld_growth_displ", dofbased, field->problem()->num_dim());

  // additional output for HDG
  writer_->WriteResult("velnp_hdg", "velocity", nodebased, field->problem()->num_dim());
  writer_->WriteResult("pressure_hdg", "pressure", nodebased, 1);
  writer_->WriteResult("tracevel_hdg", "trace_velocity", nodebased, field->problem()->num_dim());


  // additional output for XFluid level-set boundaries: write combined level-set field created from
  // all level-set coupling objects
  writer_->WriteResult("fluid_levelset_boundary", "combined_ls_bound", nodebased, 1);

  // additional output for XFluid level-set boundaries: write single level-set fields for all
  // level-set coupling objects
  int num_levelsets = 20;

  for (int k = 0; k < num_levelsets; k++)
  {
    std::ostringstream temp;
    temp << k;
    std::string name = "phinp_" + temp.str();
    writer_->WriteResult(name, name, nodebased, 1);
  }

  WriteElementResults(field);
  WriteNodeResults(field);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void XFluidFilter::WriteAllResults(PostField* field)
{
  // XFEM has changing number of degrees of freedoms
  // - restart vectors are of changing size
  // - output vectors (e.g. Paraview) have fixed size with 4 DOF per node)
  //   and are named "*_smoothed" for this reason (no integration cells)
  // calling both at the same time will crash, since restart vectors do not fit
  // the 4 DOF per node pattern.

  writer_->WriteResult(
      "velocity_smoothed", "velocity_smoothed", dofbased, field->problem()->num_dim());
  writer_->WriteResult("pressure_smoothed", "pressure_smoothed", dofbased, 1);
  writer_->WriteResult(
      "averaged_velnp", "averaged_velocity", dofbased, field->problem()->num_dim());
  writer_->WriteResult("averaged_pressure", "averaged_pressure", dofbased, 1);
  writer_->WriteResult("fsvelocity", "fsvelocity", dofbased, field->problem()->num_dim());

  WriteElementResults(field);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void InterfaceFilter::WriteAllResults(PostField* field)
{
  writer_->WriteResult("idispnp", "idispnp", dofbased, field->problem()->num_dim());
  writer_->WriteResult("idispn", "idispn", dofbased, field->problem()->num_dim());
  writer_->WriteResult("ivelnp", "ivelnp", dofbased, field->problem()->num_dim());
  writer_->WriteResult("iveln", "iveln", dofbased, field->problem()->num_dim());
  writer_->WriteResult("ivelnm", "ivelnm", dofbased, field->problem()->num_dim());
  writer_->WriteResult("iaccn", "iaccn", dofbased, field->problem()->num_dim());
  writer_->WriteResult("itrueresnp", "itrueresnp", dofbased, field->problem()->num_dim());
  WriteElementResults(field);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void AleFilter::WriteAllResults(PostField* field)
{
  writer_->WriteResult("dispnp", "displacement", dofbased, field->problem()->num_dim());
  writer_->WriteResult("det_j", "det_j", elementbased, 1);
  writer_->WriteResult("element_quality", "element_quality", elementbased, 1);
  WriteElementResults(field);
}


/*----------------------------------------------------------------------*
|                                                           wirtz 11/15 |
\*----------------------------------------------------------------------*/
void LubricationFilter::WriteAllResults(PostField* field)
{
  writer_->WriteResult("prenp", "pressure", dofbased, 1);
  writer_->WriteResult("height", "height", dofbased, 1);
  writer_->WriteResult("no_gap_DBC", "no_gap_DBC", dofbased, 1);
  writer_->WriteResult("dispnp", "displacement", nodebased, 3);

  writer_->WriteResult("viscosity", "viscosity", dofbased, 1);

  // write element results (e.g. element owner)
  WriteElementResults(field);
}

/*----------------------------------------------------------------------*
|                                                           vuong 08/16 |
\*----------------------------------------------------------------------*/
void PoroFluidMultiPhaseFilter::WriteAllResults(PostField* field)
{
  using namespace FourC;

  // compute maximum number of dofs per node on poro fluid discretization
  const DRT::Discretization& discret = *field->discretization();
  int mynumdofpernode(-1);
  for (int inode = 0; inode < discret.NumMyRowNodes(); ++inode)
  {
    const int numdof = discret.NumDof(discret.lRowNode(inode));
    if (numdof > mynumdofpernode) mynumdofpernode = numdof;
  }
  int numdofpernode(-1);
  discret.Comm().MaxAll(&mynumdofpernode, &numdofpernode, 1);

  // write results for each transported scalar
  for (int k = 1; k <= numdofpernode; k++)
  {
    std::ostringstream temp;
    temp << k;
    // write generic degree of freedom
    writer_->WriteResult("phinp_fluid", "phi_" + temp.str(), dofbased, 1, k - 1);
    // write pressure solution
    writer_->WriteResult("pressure", "pressure_" + temp.str(), dofbased, 1, k - 1);
    // write saturation solution
    writer_->WriteResult("saturation", "saturation_" + temp.str(), dofbased, 1, k - 1);

    // write generic degree of freedom
    writer_->WriteResult("phidtnp", "phidt_" + temp.str(), dofbased, 1, k - 1);

    // write flux vectors (always 3D)
    writer_->WriteResult("flux_" + temp.str(), "flux_" + temp.str(), nodebased, 3);
  }
  // write solid pressure solution
  writer_->WriteResult("solidpressure", "solid_pressure", nodebased, 1);

  // write porosity
  writer_->WriteResult("porosity", "porosity", nodebased, 1);

  // write displacement field
  writer_->WriteResult("dispnp", "ale-displacement", nodebased, field->problem()->num_dim());

  // write element results (e.g. element owner)
  WriteElementResults(field);
}

/*----------------------------------------------------------------------*
|                                                           gjb 12/07   |
\*----------------------------------------------------------------------*/
void ScaTraFilter::WriteAllResults(PostField* field)
{
  using namespace FourC;

  // compute maximum number of dofs per node on scatra discretization
  const DRT::Discretization& discret = *field->discretization();
  int mynumdofpernode(-1);
  for (int inode = 0; inode < discret.NumMyRowNodes(); ++inode)
  {
    const int numdof = discret.NumDof(discret.lRowNode(inode));
    if (numdof > mynumdofpernode) mynumdofpernode = numdof;
  }
  int numdofpernode(-1);
  discret.Comm().MaxAll(&mynumdofpernode, &numdofpernode, 1);

  // write results for each transported scalar
  for (int k = 1; k <= numdofpernode; k++)
  {
    std::ostringstream temp;
    temp << k;
    std::string name = "phi_" + temp.str();
    writer_->WriteResult("phinp", name, dofbased, 1, k - 1, true);
    writer_->WriteResult("averaged_phinp", "averaged_" + name, dofbased, 1, k - 1);
    // additional fields for meshfree problems
    writer_->WriteResult("phiatmeshfreenodes", name + "_atnodes", dofbased, 1, k - 1);
    // intermediate work-around for nurbs discretizations (no normal vectors applied)
    writer_->WriteResult("normalflux", "normalflux" + name, dofbased, 1, k - 1);
    // write flux vectors (always 3D)
    writer_->WriteResult("flux_domain_" + name, "flux_domain_" + name, nodebased, 3);
    writer_->WriteResult("flux_boundary_" + name, "flux_boundary_" + name, nodebased, 3);
    // write analytic solution
    std::string nameAnalytic = "phiAnalytic_" + temp.str();
    writer_->WriteResult("phiAnalytic", nameAnalytic, dofbased, 1, k - 1, true);
  }

  writer_->WriteResult("activation_time_np", "act_time", dofbased, 1);

  // write velocity field
  writer_->WriteResult("convec_velocity", "velocity", nodebased, field->problem()->num_dim());

  // write displacement field
  writer_->WriteResult("dispnp", "ale-displacement", nodebased, field->problem()->num_dim());

  // write optimization field
  writer_->WriteResult("x_mma_n", "optimization_variable", nodebased, 1);
  writer_->WriteResult("x_mma_e", "optimization_variable", elementbased, 1);

  // additional output for biofilm problems
  writer_->WriteResult("scfld_growth_displ", "scfld_growth_displ", nodebased, 3);
  writer_->WriteResult("scstr_growth_displ", "scstr_growth_displ", nodebased, 3);


  writer_->WriteResult("rea_coeff", "rea_coeff", elementbased, 1);
  writer_->WriteResult("diff_coeff", "diff_coeff", elementbased, 1);
  writer_->WriteResult("k_1", "k_1", elementbased, 1);
  writer_->WriteResult("k_2", "k_2", elementbased, 1);
  writer_->WriteResult("k_3", "k_3", elementbased, 1);
  writer_->WriteResult("k_4", "k_4", elementbased, 1);
  writer_->WriteResult("k_5", "k_5", elementbased, 1);
  writer_->WriteResult("k_6", "k_6", elementbased, 1);
  writer_->WriteResult("k_7", "k_7", elementbased, 1);
  writer_->WriteResult("k_8", "k_8", elementbased, 1);
  writer_->WriteResult("k_9", "k_9", elementbased, 1);
  writer_->WriteResult("k_10", "k_10", elementbased, 1);
  writer_->WriteResult("concentrationsum", "concentrationsum", elementbased, 1);

  // additional output for scatra HDG
  writer_->WriteResult("phi_hdg", "phi", nodebased, 1);
  writer_->WriteResult("tracephi_hdg", "trace_phi", nodebased, 1);
  writer_->WriteResult("gradphi_hdg", "grad_phi", nodebased, 3);
  writer_->WriteResult("activation_time_np_hdg", "act_time", nodebased, 1);
  writer_->WriteResult("ionic_currents_hdg", "ionic_currents", elementbased, 3);
  writer_->WriteResult("ionic_current_hdg_1", "ionic_current_1", elementbased, 1);
  writer_->WriteResult("ionic_current_hdg_2", "ionic_current_2", elementbased, 1);
  writer_->WriteResult("ionic_current_hdg_3", "ionic_current_3", elementbased, 1);

  // oxygen output for poromultiphase-scatra problems with artery coupling
  writer_->WriteResult("oxypartpress", "oxypartpress", nodebased, 1);

  writer_->WriteResult("micro_conc", "micro_conc", nodebased, 1);

  // write element results (e.g. element owner)
  WriteElementResults(field);
}


/*----------------------------------------------------------------------*
 | write results of electrochemistry simulation              fang 08/17 |
 *----------------------------------------------------------------------*/
void ElchFilter::WriteAllResults(PostField* field)
{
  // extract numbers of dofs per node on scatra discretization
  const DRT::Discretization& discret = *field->discretization();
  std::set<int> mynumdofpernodeset;
  for (int inode = 0; inode < discret.NumMyRowNodes(); ++inode)
    mynumdofpernodeset.insert(discret.NumDof(discret.lRowNode(inode)));
  int mysize(mynumdofpernodeset.size()), maxsize(-1);
  discret.Comm().MaxAll(&mysize, &maxsize, 1);
  std::vector<int> mynumdofpernodevec(mynumdofpernodeset.begin(), mynumdofpernodeset.end());
  mynumdofpernodevec.resize(maxsize, -1);
  std::vector<int> numdofpernodevec(maxsize * discret.Comm().NumProc(), -1);
  discret.Comm().GatherAll(
      mynumdofpernodevec.data(), numdofpernodevec.data(), mynumdofpernodevec.size());
  std::set<int> numdofpernodeset;
  for (unsigned i = 0; i < numdofpernodevec.size(); ++i)
    if (numdofpernodevec[i] > 0) numdofpernodeset.insert(numdofpernodevec[i]);

  // safety check
  switch (numdofpernodeset.size())
  {
    case 1:
    {
      // do nothing
      break;
    }

    case 2:
    {
      // check numbers
      if (*numdofpernodeset.begin() != 2 or *numdofpernodeset.rbegin() != 3)
        FOUR_C_THROW("Invalid numbers of dofs per node!");
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid numbers of dofs per node!");
      break;
    }
  }

  // extract minimum number of dofs per node
  const int numdofpernode = *numdofpernodeset.begin();

  // write results for each transported scalar
  for (int k = 1; k < numdofpernode; k++)
  {
    std::ostringstream temp;
    temp << k;
    std::string name = "c_" + temp.str();
    writer_->WriteResult("phinp", name, dofbased, 1, k - 1);
    // write flux vectors (always 3D)
    writer_->WriteResult("flux_domain_phi_" + temp.str(), "flux_domain_" + name, nodebased, 3);
    writer_->WriteResult("flux_boundary_phi_" + temp.str(), "flux_boundary_" + name, nodebased, 3);

    // temporal mean field from turbulent statistics (if present)
    writer_->WriteResult("averaged_phinp", "averaged_" + name, dofbased, 1, k - 1);
  }
  // finally, handle the electric potentials
  writer_->WriteResult("phinp", "phi", dofbased, 1, numdofpernode - 1);
  if (numdofpernodeset.size() == 2) writer_->WriteResult("phinp", "phi_ed", dofbased, 1, 2, true);

  // write flux vectors (always 3D)
  std::ostringstream temp;
  temp << numdofpernode;
  writer_->WriteResult("flux_domain_phi_" + temp.str(), "current_domain", nodebased, 3);
  writer_->WriteResult("flux_boundary_phi_" + temp.str(), "current_boundary", nodebased, 3);

  // temporal mean field from turbulent statistics (if present)
  writer_->WriteResult("averaged_phinp", "averaged_phi", dofbased, 1, numdofpernode - 1);

  // write velocity field (always 3D)
  writer_->WriteResult("convec_velocity", "velocity", nodebased, 3);

  // write displacement field (always 3D)
  writer_->WriteResult("dispnp", "ale-displacement", nodebased, 3);

  // write scatra-scatra interface layer thickness
  writer_->WriteResult("intlayerthickness", "thickness_plating", nodebased, 1);

  writer_->WriteResult("micro_conc", "micro_conc", nodebased, 1);

  // write element results (e.g. element owner)
  WriteElementResults(field);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ThermoFilter::WriteAllResults(PostField* field)
{
  // number of dofs per node in thermal problems is always 1
  const int numdofpernode = 1;

  // write temperature
  writer_->WriteResult("temperature", "temperature", dofbased, numdofpernode);

  // write temperature rate
  // writer_->WriteResult("rate", "rate", dofbased, numdofpernode);

  if (heatfluxtype_ != "none")
  {
    // although appearing here twice, only one function call to PostHeatflux
    // is really postprocessing Gauss point heatfluxes, since only _either_
    // Current _or_ Initial heatfluxes are written during simulation!
    PostHeatflux("gauss_current_heatfluxes_xyz", heatfluxtype_);
    PostHeatflux("gauss_initial_heatfluxes_xyz", heatfluxtype_);
    writer_->WriteResult("heatflux", "heatflux", nodebased, field->problem()->num_dim());
  }
  if (tempgradtype_ != "none")
  {
    // although appearing here twice, only one function call to PostHeatflux
    // is really postprocessing Gauss point temperature gradients, since only _either_
    // Initial _or_ Current temperature gradients are written during simulation!
    PostHeatflux("gauss_current_tempgrad_xyz", tempgradtype_);
    PostHeatflux("gauss_initial_tempgrad_xyz", tempgradtype_);
    writer_->WriteResult("tempgrad", "tempgrad", nodebased, field->problem()->num_dim());
  }

  // write displacement field
  writer_->WriteResult("displacement", "displacement", nodebased, field->problem()->num_dim());

  // special infomation for SLM
  writer_->WriteResult("phase", "phase", dofbased, numdofpernode);
  writer_->WriteResult("conductivity", "conductivity", dofbased, numdofpernode);
  writer_->WriteResult("capacity", "capacity", dofbased, numdofpernode);

  // write element results (e.g. element owner)
  WriteElementResults(field);

}  // ThermoFilter::WriteAllResults

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ElemagFilter::WriteAllResults(PostField* field)
{
  writer_->WriteResult("electric", "electric", nodebased, 3);
  writer_->WriteResult("electric_post", "electric_post", nodebased, 3);
  writer_->WriteResult("magnetic", "magnetic", nodebased, 3);
  writer_->WriteResult("trace", "trace", nodebased, 3);
  writer_->WriteResult("dft", "dft", nodebased, 3);
  writer_->WriteResult("conductivity", "conductivity", elementbased, 1);
  writer_->WriteResult("permittivity", "permittivity", elementbased, 1);
  writer_->WriteResult("permeability", "permeability", elementbased, 1);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void AnyFilter::WriteAllResults(PostField* field)
{
  WriteDofResults(field);
  WriteNodeResults(field);
  WriteElementResults(field);
}

FOUR_C_NAMESPACE_CLOSE