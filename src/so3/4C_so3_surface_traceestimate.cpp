/*----------------------------------------------------------------------*/
/*! \file

\brief evaluation for nitsche trace inequality estimate


\level 3
*----------------------------------------------------------------------*/

#include "4C_discretization_fem_general_utils_boundary_integration.hpp"
#include "4C_discretization_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_discretization_fem_general_utils_gausspoints.hpp"
#include "4C_discretization_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_lib_element_integration_select.hpp"
#include "4C_linalg_utils_densematrix_determinant.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_mat_fourieriso.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_nurbs_discret.hpp"
#include "4C_so3_surface.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |                                                           seitz 11/16|
 *----------------------------------------------------------------------*/
double DRT::ELEMENTS::StructuralSurface::estimate_nitsche_trace_max_eigenvalue_combined(
    std::vector<double>& parent_disp)
{
  switch (parent_element()->Shape())
  {
    case CORE::FE::CellType::hex8:
      if (Shape() == CORE::FE::CellType::quad4)
        return estimate_nitsche_trace_max_eigenvalue_combined<CORE::FE::CellType::hex8,
            CORE::FE::CellType::quad4>(parent_disp);
      else
        FOUR_C_THROW("how can an hex8 element have a surface that is not quad4 ???");
      break;
    case CORE::FE::CellType::hex27:
      return estimate_nitsche_trace_max_eigenvalue_combined<CORE::FE::CellType::hex27,
          CORE::FE::CellType::quad9>(parent_disp);
      break;
    case CORE::FE::CellType::tet4:
      return estimate_nitsche_trace_max_eigenvalue_combined<CORE::FE::CellType::tet4,
          CORE::FE::CellType::tri3>(parent_disp);
      break;
    case CORE::FE::CellType::nurbs27:
      return estimate_nitsche_trace_max_eigenvalue_combined<CORE::FE::CellType::nurbs27,
          CORE::FE::CellType::nurbs9>(parent_disp);
      break;
    default:
      FOUR_C_THROW("parent shape not implemented");
  }

  return 0;
}


template <CORE::FE::CellType dt_vol, CORE::FE::CellType dt_surf>
double DRT::ELEMENTS::StructuralSurface::estimate_nitsche_trace_max_eigenvalue_combined(
    std::vector<double>& parent_disp)
{
  const int dim = CORE::FE::dim<dt_vol>;
  const int num_dof = CORE::FE::num_nodes<dt_vol> * CORE::FE::dim<dt_vol>;
  const int dim_image = CORE::FE::num_nodes<dt_vol> * CORE::FE::dim<dt_vol> -
                        CORE::FE::dim<dt_vol> * (CORE::FE::dim<dt_vol> + 1) / 2;

  CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3> xrefe;
  CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3> xcurr;

  for (int i = 0; i < parent_element()->num_node(); ++i)
    for (int d = 0; d < dim; ++d)
    {
      xrefe(i, d) = parent_element()->Nodes()[i]->X()[d];
      xcurr(i, d) = xrefe(i, d) + parent_disp[i * dim + d];
    }

  CORE::LINALG::Matrix<num_dof, num_dof> vol, surf;

  trace_estimate_vol_matrix<dt_vol>(xrefe, xcurr, vol);
  trace_estimate_surf_matrix<dt_vol, dt_surf>(xrefe, xcurr, surf);

  CORE::LINALG::Matrix<num_dof, dim_image> proj, tmp;
  subspace_projector<dt_vol>(xcurr, proj);

  CORE::LINALG::Matrix<dim_image, dim_image> vol_red, surf_red;

  tmp.Multiply(vol, proj);
  vol_red.MultiplyTN(proj, tmp);
  tmp.Multiply(surf, proj);
  surf_red.MultiplyTN(proj, tmp);

  CORE::LINALG::SerialDenseMatrix vol_red_sd(
      Teuchos::View, vol_red.A(), dim_image, dim_image, dim_image);
  CORE::LINALG::SerialDenseMatrix surf_red_sd(
      Teuchos::View, surf_red.A(), dim_image, dim_image, dim_image);

  return CORE::LINALG::GeneralizedEigen(surf_red_sd, vol_red_sd);
}

template <CORE::FE::CellType dt_vol>
void DRT::ELEMENTS::StructuralSurface::trace_estimate_vol_matrix(
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xrefe,
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xcurr,
    CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol> * 3, CORE::FE::num_nodes<dt_vol> * 3>& vol)
{
  const int dim = CORE::FE::dim<dt_vol>;

  double jac;
  CORE::LINALG::Matrix<3, 3> defgrd;
  CORE::LINALG::Matrix<3, 3> rcg;
  CORE::LINALG::Matrix<6, 1> glstrain;
  CORE::LINALG::Matrix<6, CORE::FE::num_nodes<dt_vol> * 3> bop;
  CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol> * 3, 6> bc;
  CORE::LINALG::Matrix<dim, CORE::FE::num_nodes<dt_vol>> N_XYZ;

  CORE::FE::IntPointsAndWeights<dim> ip(DRT::ELEMENTS::DisTypeToOptGaussRule<dt_vol>::rule);

  for (int gp = 0; gp < ip.IP().nquad; ++gp)
  {
    const CORE::LINALG::Matrix<3, 1> xi(ip.IP().qxg[gp], false);
    strains<dt_vol>(xrefe, xcurr, xi, jac, defgrd, glstrain, rcg, bop, N_XYZ);

    CORE::LINALG::Matrix<6, 6> cmat(true);
    CORE::LINALG::Matrix<6, 1> stress(true);
    Teuchos::ParameterList params;
    Teuchos::rcp_dynamic_cast<MAT::So3Material>(parent_element()->Material())
        ->Evaluate(&defgrd, &glstrain, params, &stress, &cmat, gp, parent_element()->Id());
    bc.MultiplyTN(bop, cmat);
    vol.Multiply(ip.IP().qwgt[gp] * jac, bc, bop, 1.);
  }

  return;
}


template <CORE::FE::CellType dt_vol, CORE::FE::CellType dt_surf>
void DRT::ELEMENTS::StructuralSurface::trace_estimate_surf_matrix(
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xrefe,
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xcurr,
    CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol> * 3, CORE::FE::num_nodes<dt_vol> * 3>& surf)
{
  const int dim = CORE::FE::dim<dt_vol>;

  CORE::LINALG::Matrix<6, 6> id4;
  for (int i = 0; i < 3; ++i) id4(i, i) = 1.;
  for (int i = 3; i < 6; ++i) id4(i, i) = 2.;

  CORE::LINALG::SerialDenseMatrix xrefe_surf(CORE::FE::num_nodes<dt_surf>, dim);
  material_configuration(xrefe_surf);

  std::vector<double> n(3);
  CORE::LINALG::Matrix<3, 1> n_v(n.data(), true);
  CORE::LINALG::Matrix<3, 3> nn;
  double detA;
  double jac;
  CORE::LINALG::Matrix<3, 3> defgrd;
  CORE::LINALG::Matrix<3, 3> rcg;
  CORE::LINALG::Matrix<6, 1> glstrain;
  CORE::LINALG::Matrix<6, CORE::FE::num_nodes<dt_vol> * 3> bop;
  CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol> * 3, 6> bc;
  CORE::LINALG::Matrix<dim, CORE::FE::num_nodes<dt_vol>> N_XYZ;

  CORE::FE::IntPointsAndWeights<dim - 1> ip(DRT::ELEMENTS::DisTypeToOptGaussRule<dt_surf>::rule);
  CORE::LINALG::SerialDenseMatrix deriv_surf(2, CORE::FE::num_nodes<dt_surf>);

  for (int gp = 0; gp < ip.IP().nquad; ++gp)
  {
    CORE::FE::CollectedGaussPoints intpoints =
        CORE::FE::CollectedGaussPoints(1);  // reserve just for 1 entry ...
    intpoints.Append(ip.IP().qxg[gp][0], ip.IP().qxg[gp][1], 0.0, ip.IP().qwgt[gp]);

    // get coordinates of gauss point w.r.t. local parent coordinate system
    CORE::LINALG::SerialDenseMatrix pqxg(1, 3);
    CORE::LINALG::Matrix<3, 3> derivtrafo;

    CORE::FE::BoundaryGPToParentGP<3>(
        pqxg, derivtrafo, intpoints, parent_element()->Shape(), Shape(), FaceParentNumber());

    CORE::LINALG::Matrix<3, 1> xi;
    for (int i = 0; i < 3; ++i) xi(i) = pqxg(0, i);
    strains<dt_vol>(xrefe, xcurr, xi, jac, defgrd, glstrain, rcg, bop, N_XYZ);

    CORE::LINALG::Matrix<6, 6> cmat(true);
    CORE::LINALG::Matrix<6, 1> stress(true);
    Teuchos::ParameterList params;
    Teuchos::rcp_dynamic_cast<MAT::So3Material>(parent_element()->Material())
        ->Evaluate(&defgrd, &glstrain, params, &stress, &cmat, gp, parent_element()->Id());

    double normalfac = 1.;
    if (Shape() == CORE::FE::CellType::nurbs9)
    {
      std::vector<CORE::LINALG::SerialDenseVector> parentknots(dim);
      std::vector<CORE::LINALG::SerialDenseVector> boundaryknots(dim - 1);
      dynamic_cast<DRT::NURBS::NurbsDiscretization*>(
          GLOBAL::Problem::Instance()->GetDis("structure").get())
          ->GetKnotVector()
          ->get_boundary_ele_and_parent_knots(
              parentknots, boundaryknots, normalfac, parent_element()->Id(), FaceParentNumber());

      CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_surf>, 1> weights, shapefcn;
      for (int i = 0; i < CORE::FE::num_nodes<dt_surf>; ++i)
        weights(i) = dynamic_cast<DRT::NURBS::ControlPoint*>(Nodes()[i])->W();

      CORE::LINALG::Matrix<2, 1> xi_surf;
      xi_surf(0) = ip.IP().qxg[gp][0];
      xi_surf(1) = ip.IP().qxg[gp][1];
      CORE::FE::NURBS::nurbs_get_2D_funct_deriv(
          shapefcn, deriv_surf, xi_surf, boundaryknots, weights, dt_surf);
    }
    else
      CORE::FE::shape_function_2D_deriv1(
          deriv_surf, ip.IP().qxg[gp][0], ip.IP().qxg[gp][1], Shape());

    surface_integration(detA, n, xrefe_surf, deriv_surf);
    n_v.Scale(normalfac);
    n_v.Scale(1. / n_v.Norm2());
    nn.MultiplyNT(n_v, n_v);

    CORE::LINALG::Matrix<6, 6> cn;
    MAT::AddSymmetricHolzapfelProduct(cn, rcg, nn, .25);

    CORE::LINALG::Matrix<6, 6> tmp1, tmp2;
    tmp1.Multiply(cmat, id4);
    tmp2.Multiply(tmp1, cn);
    tmp1.Multiply(tmp2, id4);
    tmp2.Multiply(tmp1, cmat);

    CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol> * 3, 6> tmp3;
    tmp3.MultiplyTN(bop, tmp2);

    surf.Multiply(detA * ip.IP().qwgt[gp], tmp3, bop, 1.);
  }

  return;
}

template <CORE::FE::CellType dt_vol>
void DRT::ELEMENTS::StructuralSurface::strains(
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xrefe,
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xcurr,
    const CORE::LINALG::Matrix<3, 1>& xi, double& jac, CORE::LINALG::Matrix<3, 3>& defgrd,
    CORE::LINALG::Matrix<6, 1>& glstrain, CORE::LINALG::Matrix<3, 3>& rcg,
    CORE::LINALG::Matrix<6, CORE::FE::num_nodes<dt_vol> * 3>& bop,
    CORE::LINALG::Matrix<3, CORE::FE::num_nodes<dt_vol>>& N_XYZ)
{
  const int dim = CORE::FE::dim<dt_vol>;
  const int num_node = CORE::FE::num_nodes<dt_vol>;
  CORE::LINALG::Matrix<dim, num_node> deriv;

  if (dt_vol == CORE::FE::CellType::nurbs27)
  {
    std::vector<CORE::LINALG::SerialDenseVector> knots;
    dynamic_cast<DRT::NURBS::NurbsDiscretization*>(
        GLOBAL::Problem::Instance()->GetDis("structure").get())
        ->GetKnotVector()
        ->GetEleKnots(knots, ParentElementId());

    CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 1> weights, shapefcn;

    for (int i = 0; i < CORE::FE::num_nodes<dt_vol>; ++i)
      weights(i) = dynamic_cast<DRT::NURBS::ControlPoint*>(parent_element()->Nodes()[i])->W();

    CORE::FE::NURBS::nurbs_get_3D_funct_deriv(shapefcn, deriv, xi, knots, weights, dt_vol);
  }
  else
    CORE::FE::shape_function_deriv1<dt_vol>(xi, deriv);

  CORE::LINALG::Matrix<dim, dim> invJ;
  invJ.Multiply(deriv, xrefe);
  jac = invJ.Invert();
  N_XYZ.Multiply(invJ, deriv);
  defgrd.MultiplyTT(xcurr, N_XYZ);

  rcg.MultiplyTN(defgrd, defgrd);
  glstrain(0) = 0.5 * (rcg(0, 0) - 1.0);
  glstrain(1) = 0.5 * (rcg(1, 1) - 1.0);
  glstrain(2) = 0.5 * (rcg(2, 2) - 1.0);
  glstrain(3) = rcg(0, 1);
  glstrain(4) = rcg(1, 2);
  glstrain(5) = rcg(2, 0);

  for (int i = 0; i < num_node; ++i)
  {
    bop(0, dim * i + 0) = defgrd(0, 0) * N_XYZ(0, i);
    bop(0, dim * i + 1) = defgrd(1, 0) * N_XYZ(0, i);
    bop(0, dim * i + 2) = defgrd(2, 0) * N_XYZ(0, i);
    bop(1, dim * i + 0) = defgrd(0, 1) * N_XYZ(1, i);
    bop(1, dim * i + 1) = defgrd(1, 1) * N_XYZ(1, i);
    bop(1, dim * i + 2) = defgrd(2, 1) * N_XYZ(1, i);
    bop(2, dim * i + 0) = defgrd(0, 2) * N_XYZ(2, i);
    bop(2, dim * i + 1) = defgrd(1, 2) * N_XYZ(2, i);
    bop(2, dim * i + 2) = defgrd(2, 2) * N_XYZ(2, i);
    /* ~~~ */
    bop(3, dim * i + 0) = defgrd(0, 0) * N_XYZ(1, i) + defgrd(0, 1) * N_XYZ(0, i);
    bop(3, dim * i + 1) = defgrd(1, 0) * N_XYZ(1, i) + defgrd(1, 1) * N_XYZ(0, i);
    bop(3, dim * i + 2) = defgrd(2, 0) * N_XYZ(1, i) + defgrd(2, 1) * N_XYZ(0, i);
    bop(4, dim * i + 0) = defgrd(0, 1) * N_XYZ(2, i) + defgrd(0, 2) * N_XYZ(1, i);
    bop(4, dim * i + 1) = defgrd(1, 1) * N_XYZ(2, i) + defgrd(1, 2) * N_XYZ(1, i);
    bop(4, dim * i + 2) = defgrd(2, 1) * N_XYZ(2, i) + defgrd(2, 2) * N_XYZ(1, i);
    bop(5, dim * i + 0) = defgrd(0, 2) * N_XYZ(0, i) + defgrd(0, 0) * N_XYZ(2, i);
    bop(5, dim * i + 1) = defgrd(1, 2) * N_XYZ(0, i) + defgrd(1, 0) * N_XYZ(2, i);
    bop(5, dim * i + 2) = defgrd(2, 2) * N_XYZ(0, i) + defgrd(2, 0) * N_XYZ(2, i);
  }

  return;
}


template <CORE::FE::CellType dt_vol>
void DRT::ELEMENTS::StructuralSurface::subspace_projector(
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xcurr,
    CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol> * CORE::FE::dim<dt_vol>,
        CORE::FE::num_nodes<dt_vol> * CORE::FE::dim<dt_vol> -
            CORE::FE::dim<dt_vol>*(CORE::FE::dim<dt_vol> + 1) / 2>& proj)
{
  const int dim = CORE::FE::dim<dt_vol>;
  const int num_node = CORE::FE::num_nodes<dt_vol>;
  if (dim != 3) FOUR_C_THROW("this should be 3D");

  CORE::LINALG::Matrix<3, 1> c;
  for (int r = 0; r < (int)xcurr.numRows(); ++r)
    for (int d = 0; d < (int)xcurr.numCols(); ++d) c(d) += xcurr(r, d);
  c.Scale(1. / xcurr.numRows());

  CORE::LINALG::Matrix<dim, 1> r[3];
  for (int i = 0; i < 3; ++i) r[i](i) = 1.;

  // basis, where the first six entries are the rigid body modes and the
  // remaining are constructed to be orthogonal to the rigid body modes
  CORE::LINALG::Matrix<dim * num_node, 1> basis[dim * num_node];

  // rigid body translations
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < num_node; ++j) basis[i](j * dim + i) = 1.;

  // rigid body rotations
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < num_node; ++j)
    {
      CORE::LINALG::Matrix<3, 1> x;
      for (int d = 0; d < 3; ++d) x(d) = xcurr(j, d);
      x.Update(-1., c, 1.);
      CORE::LINALG::Matrix<3, 1> cross;
      cross.CrossProduct(r[i], x);
      for (int k = 0; k < 3; ++k) basis[i + 3](j * 3 + k) = cross(k);
    }
  for (int i = 0; i < 6; ++i) basis[i].Scale(1. / basis[i].Norm2());

  // build the remaining basis vectors by generalized cross products
  for (int i = 6; i < dim * num_node; ++i)
  {
    double sign = +1.;
    int off = 0;
    bool new_basis_found = false;
    for (off = 0; (off < dim * num_node - i) && !new_basis_found; ++off)
    {
      for (int j = 0; j < i + 1; ++j)
      {
        CORE::LINALG::SerialDenseMatrix det(i, i, true);
        for (int c = 0; c < i; ++c)
        {
          for (int k = 0; k < j; ++k) det(k, c) = basis[c](k + off);
          for (int k = j; k < i; ++k) det(k, c) = basis[c](k + 1 + off);
        }
        basis[i](j + off) = CORE::LINALG::DeterminantLU(det) * sign;
        sign *= -1.;
      }
      if (basis[i].Norm2() > 1.e-6)
      {
        basis[i].Scale(1. / basis[i].Norm2());
        new_basis_found = true;
      }
    }
    if (!new_basis_found) FOUR_C_THROW("no new basis vector found");
  }

  // at this point basis should already contain an ONB.
  // due to cut-off errors we do another sweep of Gram-Schmidt
  for (int i = 0; i < dim * num_node; ++i)
  {
    const CORE::LINALG::Matrix<dim * num_node, 1> tmp(basis[i]);
    for (int j = 0; j < i; ++j) basis[i].Update(-tmp.Dot(basis[j]), basis[j], 1.);

    basis[i].Scale(1. / basis[i].Norm2());
  }

  // hand out the projection matrix, i.e. the ONB not containing rigid body modes
  for (int i = 0; i < dim * num_node; ++i)
    for (int j = 6; j < dim * num_node; ++j) proj(i, j - 6) = basis[j](i);
}



/*----------------------------------------------------------------------*
 |                                                           seitz 11/16|
 *----------------------------------------------------------------------*/
double DRT::ELEMENTS::StructuralSurface::estimate_nitsche_trace_max_eigenvalue_tsi(
    std::vector<double>& parent_disp)
{
  switch (parent_element()->Shape())
  {
    case CORE::FE::CellType::hex8:
      if (Shape() == CORE::FE::CellType::quad4)
        return estimate_nitsche_trace_max_eigenvalue_tsi<CORE::FE::CellType::hex8,
            CORE::FE::CellType::quad4>(parent_disp);
      else
        FOUR_C_THROW("how can an hex8 element have a surface that is not quad4 ???");
      break;
    case CORE::FE::CellType::hex27:
      return estimate_nitsche_trace_max_eigenvalue_tsi<CORE::FE::CellType::hex27,
          CORE::FE::CellType::quad9>(parent_disp);
    case CORE::FE::CellType::tet4:
      return estimate_nitsche_trace_max_eigenvalue_tsi<CORE::FE::CellType::tet4,
          CORE::FE::CellType::tri3>(parent_disp);
    case CORE::FE::CellType::nurbs27:
      return estimate_nitsche_trace_max_eigenvalue_tsi<CORE::FE::CellType::nurbs27,
          CORE::FE::CellType::nurbs9>(parent_disp);
    default:
      FOUR_C_THROW("parent shape not implemented");
  }

  return 0;
}

template <CORE::FE::CellType dt_vol, CORE::FE::CellType dt_surf>
double DRT::ELEMENTS::StructuralSurface::estimate_nitsche_trace_max_eigenvalue_tsi(
    std::vector<double>& parent_disp)
{
  const int dim = CORE::FE::dim<dt_vol>;
  const int num_dof = CORE::FE::num_nodes<dt_vol>;
  const int dim_image = CORE::FE::num_nodes<dt_vol> - 1;

  CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3> xrefe;
  CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3> xcurr;

  for (int i = 0; i < parent_element()->num_node(); ++i)
    for (int d = 0; d < dim; ++d)
    {
      xrefe(i, d) = parent_element()->Nodes()[i]->X()[d];
      xcurr(i, d) = xrefe(i, d) + parent_disp[i * dim + d];
    }

  CORE::LINALG::Matrix<num_dof, num_dof> vol, surf;

  trace_estimate_vol_matrix_tsi<dt_vol>(xrefe, xcurr, vol);
  trace_estimate_surf_matrix_tsi<dt_vol, dt_surf>(xrefe, xcurr, surf);


  CORE::LINALG::Matrix<num_dof, dim_image> proj, tmp;
  subspace_projector_scalar<dt_vol>(proj);

  CORE::LINALG::Matrix<dim_image, dim_image> vol_red, surf_red;

  tmp.Multiply(vol, proj);
  vol_red.MultiplyTN(proj, tmp);
  tmp.Multiply(surf, proj);
  surf_red.MultiplyTN(proj, tmp);

  CORE::LINALG::SerialDenseMatrix vol_red_sd(
      Teuchos::View, vol_red.A(), dim_image, dim_image, dim_image);
  CORE::LINALG::SerialDenseMatrix surf_red_sd(
      Teuchos::View, surf_red.A(), dim_image, dim_image, dim_image);

  return CORE::LINALG::GeneralizedEigen(surf_red_sd, vol_red_sd);
}

template <CORE::FE::CellType dt_vol>
void DRT::ELEMENTS::StructuralSurface::trace_estimate_vol_matrix_tsi(
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xrefe,
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xcurr,
    CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, CORE::FE::num_nodes<dt_vol>>& vol)
{
  const int dim = CORE::FE::dim<dt_vol>;
  const int num_node = CORE::FE::num_nodes<dt_vol>;

  double jac;
  CORE::LINALG::Matrix<3, 3> defgrd;
  CORE::LINALG::Matrix<3, 3> rcg;
  CORE::LINALG::Matrix<6, 1> glstrain;
  CORE::LINALG::Matrix<6, CORE::FE::num_nodes<dt_vol> * 3> bop;
  CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol> * 3, 6> bc;
  CORE::LINALG::Matrix<dim, num_node> N_XYZ, iC_N_XYZ;

  CORE::FE::IntPointsAndWeights<dim> ip(DRT::ELEMENTS::DisTypeToOptGaussRule<dt_vol>::rule);

  if (parent_element()->NumMaterial() < 2) FOUR_C_THROW("where's my second material");
  Teuchos::RCP<MAT::FourierIso> mat_thr =
      Teuchos::rcp_dynamic_cast<MAT::FourierIso>(parent_element()->Material(1), true);
  const double k0 = mat_thr->Conductivity();

  for (int gp = 0; gp < ip.IP().nquad; ++gp)
  {
    const CORE::LINALG::Matrix<3, 1> xi(ip.IP().qxg[gp], false);
    strains<dt_vol>(xrefe, xcurr, xi, jac, defgrd, glstrain, rcg, bop, N_XYZ);

    CORE::LINALG::Matrix<3, 3> iC;
    iC.MultiplyTN(defgrd, defgrd);
    iC.Invert();

    iC_N_XYZ.Multiply(iC, N_XYZ);
    iC_N_XYZ.Scale(k0);

    vol.MultiplyTN(ip.IP().qwgt[gp] * jac, N_XYZ, iC_N_XYZ, 1.);
  }
}


template <CORE::FE::CellType dt_vol, CORE::FE::CellType dt_surf>
void DRT::ELEMENTS::StructuralSurface::trace_estimate_surf_matrix_tsi(
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xrefe,
    const CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, 3>& xcurr,
    CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, CORE::FE::num_nodes<dt_vol>>& surf)
{
  const int dim = CORE::FE::dim<dt_vol>;
  const int num_node = CORE::FE::num_nodes<dt_vol>;

  double jac;
  CORE::LINALG::Matrix<3, 3> defgrd;
  CORE::LINALG::Matrix<3, 3> rcg;
  CORE::LINALG::Matrix<6, 1> glstrain;
  CORE::LINALG::Matrix<6, CORE::FE::num_nodes<dt_vol> * 3> bop;
  CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol> * 3, 6> bc;
  CORE::LINALG::Matrix<dim, num_node> N_XYZ;
  CORE::LINALG::Matrix<1, num_node> iCn_N_XYZ;

  CORE::LINALG::SerialDenseMatrix xrefe_surf(CORE::FE::num_nodes<dt_surf>, dim);
  material_configuration(xrefe_surf);

  std::vector<double> n(3);
  CORE::LINALG::Matrix<3, 1> n_v(n.data(), true), iCn;
  double detA;

  CORE::FE::IntPointsAndWeights<dim - 1> ip(DRT::ELEMENTS::DisTypeToOptGaussRule<dt_surf>::rule);
  CORE::LINALG::SerialDenseMatrix deriv_surf(2, CORE::FE::num_nodes<dt_surf>);

  if (parent_element()->NumMaterial() < 2) FOUR_C_THROW("where's my second material");
  Teuchos::RCP<MAT::FourierIso> mat_thr =
      Teuchos::rcp_dynamic_cast<MAT::FourierIso>(parent_element()->Material(1), true);
  const double k0 = mat_thr->Conductivity();

  for (int gp = 0; gp < ip.IP().nquad; ++gp)
  {
    CORE::FE::shape_function_2D_deriv1(deriv_surf, ip.IP().qxg[gp][0], ip.IP().qxg[gp][1], Shape());
    surface_integration(detA, n, xrefe_surf, deriv_surf);
    n_v.Scale(1. / n_v.Norm2());

    CORE::FE::CollectedGaussPoints intpoints =
        CORE::FE::CollectedGaussPoints(1);  // reserve just for 1 entry ...
    intpoints.Append(ip.IP().qxg[gp][0], ip.IP().qxg[gp][1], 0.0, ip.IP().qwgt[gp]);

    // get coordinates of gauss point w.r.t. local parent coordinate system
    CORE::LINALG::SerialDenseMatrix pqxg(1, 3);
    CORE::LINALG::Matrix<3, 3> derivtrafo;

    CORE::FE::BoundaryGPToParentGP<3>(
        pqxg, derivtrafo, intpoints, parent_element()->Shape(), Shape(), FaceParentNumber());

    CORE::LINALG::Matrix<3, 1> xi;
    for (int i = 0; i < 3; ++i) xi(i) = pqxg(0, i);

    strains<dt_vol>(xrefe, xcurr, xi, jac, defgrd, glstrain, rcg, bop, N_XYZ);

    CORE::LINALG::Matrix<3, 3> iC;
    iC.MultiplyTN(defgrd, defgrd);
    iC.Invert();
    iCn.Multiply(iC, n_v);

    iCn_N_XYZ.MultiplyTN(iCn, N_XYZ);
    iCn_N_XYZ.Scale(k0);

    surf.MultiplyTN(detA * ip.IP().qwgt[gp], iCn_N_XYZ, iCn_N_XYZ, 1.);
  }
}



template <CORE::FE::CellType dt_vol>
void DRT::ELEMENTS::StructuralSurface::subspace_projector_scalar(
    CORE::LINALG::Matrix<CORE::FE::num_nodes<dt_vol>, CORE::FE::num_nodes<dt_vol> - 1>& proj)
{
  const int num_node = CORE::FE::num_nodes<dt_vol>;
  CORE::LINALG::Matrix<num_node, 1> basis[num_node];

  for (int i = 0; i < num_node; ++i) basis[0](i) = 1.;

  for (int i = 1; i < num_node; ++i)
  {
    double sign = +1.;
    for (int j = 0; j < i + 1; ++j)
    {
      CORE::LINALG::SerialDenseMatrix det(i, i, true);
      for (int c = 0; c < i; ++c)
      {
        for (int k = 0; k < j; ++k) det(k, c) = basis[c](k);
        for (int k = j; k < i; ++k) det(k, c) = basis[c](k + 1);
      }
      basis[i](j) = CORE::LINALG::DeterminantLU(det) * sign;
      sign *= -1.;
    }
    basis[i].Scale(1. / basis[i].Norm2());
  }

  // hand out the projection matrix, i.e. the ONB not containing rigid body modes
  for (int i = 0; i < num_node; ++i)
    for (int j = 1; j < num_node; ++j) proj(i, j - 1) = basis[j](i);
}

FOUR_C_NAMESPACE_CLOSE