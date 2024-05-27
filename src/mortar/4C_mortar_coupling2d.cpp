/*-----------------------------------------------------------------------*/
/*! \file
\brief Classes for mortar coupling in 2D.

\level 2

*/
/*----------------------------------------------------------------------*/

#include "4C_mortar_coupling2d.hpp"

#include "4C_lib_discret.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_integrator.hpp"
#include "4C_mortar_node.hpp"
#include "4C_mortar_projector.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MORTAR::Coupling2d::Coupling2d(DRT::Discretization& idiscret, int dim, bool quad,
    Teuchos::ParameterList& params, MORTAR::Element& sele, MORTAR::Element& mele)
    : idiscret_(idiscret),
      dim_(dim),
      quad_(quad),
      imortar_(params),
      sele_(sele),
      mele_(mele),
      overlap_(false)
{
  // initialize variables
  hasproj_.resize(4);
  xiproj_.resize(4);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Comm& MORTAR::Coupling2d::Comm() const { return idiscret_.Comm(); }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool MORTAR::Coupling2d::rough_check_orient()
{
  // we first need the master element center
  std::array<double, 2> loccenter = {0.0, 0.0};

  // compute the unit normal vector at the slave element center
  std::array<double, 3> nsc = {0.0, 0.0, 0.0};
  SlaveElement().compute_unit_normal_at_xi(loccenter.data(), nsc.data());

  // compute the unit normal vector at the master element center
  std::array<double, 3> nmc = {0.0, 0.0, 0.0};
  MasterElement().compute_unit_normal_at_xi(loccenter.data(), nmc.data());

  // check orientation of the two normals
  double dot = nsc[0] * nmc[0] + nsc[1] * nmc[1] + nsc[2] * nmc[2];
  return (dot < -0.1);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool MORTAR::Coupling2d::Project()
{
  // initialize projection status
  hasproj_[0] = false;  // slave 0 end node
  hasproj_[1] = false;  // slave 1 end node
  hasproj_[2] = false;  // master 0 end node
  hasproj_[3] = false;  // master 1 end node

  // rough check of orientation of element centers
  // if slave and master element center normals form an
  // angle > 90 degrees the pair will not be considered further
  bool orient = rough_check_orient();
  if (!orient) return false;

  // get slave and master element nodes
  DRT::Node** mysnodes = SlaveElement().Nodes();
  if (!mysnodes) FOUR_C_THROW("IntegrateOverlap: Null pointer for mysnodes!");
  DRT::Node** mymnodes = MasterElement().Nodes();
  if (!mymnodes) FOUR_C_THROW("IntegrateOverlap: Null pointer for mymnodes!");

  // project slave nodes onto master element
  for (int i = 0; i < SlaveElement().num_node(); ++i)
  {
    auto* snode = dynamic_cast<MORTAR::Node*>(mysnodes[i]);
    std::array<double, 2> xi = {0.0, 0.0};

    if (SlaveElement().Shape() == CORE::FE::CellType::nurbs3)
    {
      std::array<double, 2> xinode = {0., 0.};
      if (i == 0)
      {
        xinode[0] = -1.;
      }
      if (i == 1)
      {
        xinode[0] = +1.;
      }

      // for nurbs we need to use the Gauss point projector, since the actual spatial coords
      // of the point to be projected is calculated by N*X using shape functions N and CP coords X
      MORTAR::Projector::Impl(SlaveElement(), mele_)
          ->ProjectGaussPoint2D(SlaveElement(), xinode.data(), mele_, xi.data());
    }
    else
    {
      // TODO random?
      MORTAR::Projector::Impl(SlaveElement())
          ->ProjectNodalNormal(*snode, MasterElement(), xi.data());
    }

    // save projection if it is feasible
    // we need an expanded feasible domain in order to check pathological
    // cases due to round-off error and iteration tolerances later!
    if ((-1.0 - MORTARPROJTOL <= xi[0]) && (xi[0] <= 1.0 + MORTARPROJTOL))
    {
      // for element overlap only the outer nodes are of interest
      if (i < 2)
      {
        hasproj_[i] = true;
        xiproj_[i] = xi[0];
      }
      // nevertheless we need the inner node projection status later (weighted gap)
      snode->HasProj() = true;
    }
  }

  // project master nodes onto slave element
  for (int i = 0; i < 2; ++i)
  {
    auto* mnode = dynamic_cast<MORTAR::Node*>(mymnodes[i]);
    std::array<double, 2> xi = {0.0, 0.0};

    if (MasterElement().Shape() == CORE::FE::CellType::nurbs3)
    {
      std::array<double, 2> xinode = {0., 0.};
      if (i == 0)
      {
        xinode[0] = -1.;
      }
      if (i == 1)
      {
        xinode[0] = +1.;
      }

      // for nurbs, we introduce a dummy mortar node at the actual spatial position of the master
      // side element boundary. Hence, we need that location
      std::vector<double> xm(2, 0.0);
      CORE::LINALG::SerialDenseVector mval(mele_.num_node());
      CORE::LINALG::SerialDenseMatrix deriv(mele_.num_node(), 1);
      mele_.EvaluateShape(xinode.data(), mval, deriv, mele_.num_node());

      for (int mn = 0; mn < MasterElement().num_node(); mn++)
      {
        auto* mnode2 = dynamic_cast<MORTAR::Node*>(mymnodes[mn]);
        for (int dim = 0; dim < 2; ++dim) xm[dim] += mval(mn) * mnode2->xspatial()[dim];
      }
      std::vector<int> mdofs(2);
      MORTAR::Node tmp_node(mnode->Id(), xm, mnode->Owner(), mdofs, false);
      MORTAR::Projector::Impl(SlaveElement())
          ->project_element_normal(tmp_node, SlaveElement(), xi.data());
    }
    else
    {
      // TODO random?
      MORTAR::Projector::Impl(SlaveElement())
          ->project_element_normal(*mnode, SlaveElement(), xi.data());
    }

    // save projection if it is feasible
    // we need an expanded feasible domain in order to check pathological
    // cases due to round-off error and iteration tolerances later!!!
    if ((-1.0 - MORTARPROJTOL <= xi[0]) && (xi[0] <= 1.0 + MORTARPROJTOL))
    {
      // for element overlap only the outer nodes are of interest
      if (i < 2)
      {
        hasproj_[i + 2] = true;
        xiproj_[i + 2] = xi[0];
      }
    }
  }

  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool MORTAR::Coupling2d::DetectOverlap()
{
  /**********************************************************************/
  /* OVERLAP CASES                                                      */
  /* Depending on mxi and sxi overlap will be decided!                  */
  /* Even for 3noded elements only the two end nodes matter in 2D!      */
  /* There are several cases how the 2 elements can overlap. Handle all */
  /* of them, including the ones that they don't overlap at all!        */
  /**********************************************************************/

  // For the non-overlapping cases, the possibility of an identical local
  // node numbering direction for both sides is taken into account!!
  // (this can happen, when elements far from each other are projected,
  // which actually should be impossible due to the search radius
  // condition in the potential coupling pair search above!
  // But you never know...)
  // For the overlapping cases, it is a prerequisite that the two local
  // node numbering directions are opposite!!
  // (this is the case, when the elements are sufficiently near each other,
  // which is ensured by only processing nodes that fulfill the
  // search radius condition above!)
  // CAUTION: The bool output variable in this method is a REAL output
  // variable, determining whether there is an overlap or not!
  // initialize local working variables
  bool overlap = false;
  double sxia = 0.0;
  double sxib = 0.0;
  double mxia = 0.0;
  double mxib = 0.0;

  // local working copies of input variables
  bool s0hasproj = hasproj_[0];
  bool s1hasproj = hasproj_[1];
  bool m0hasproj = hasproj_[2];
  bool m1hasproj = hasproj_[3];

  std::vector<double> sprojxi(2);
  sprojxi[0] = xiproj_[0];
  sprojxi[1] = xiproj_[1];

  std::vector<double> mprojxi(2);
  mprojxi[0] = xiproj_[2];
  mprojxi[1] = xiproj_[3];

  /* CASE 1 (NO OVERLAP):
   no feasible projection found for any of the 4 outer element nodes  */

  if (!s0hasproj && !s1hasproj && !m0hasproj && !m1hasproj)
  {
    // do nothing
  }

  /* CASES 2-5 (NO OVERLAP):
   feasible projection found only for 1 of the 4 outer element nodes
   (this can happen due to the necessary projection tolerance!!!)     */

  else if (s0hasproj && !s1hasproj && !m0hasproj && !m1hasproj)
  {
    if ((-1.0 + MORTARPROJTOL <= sprojxi[0]) && (sprojxi[0] <= 1.0 - MORTARPROJTOL))
    {
      std::cout << "SElement Node IDs: " << (SlaveElement().Nodes()[0])->Id() << " "
                << (SlaveElement().Nodes()[1])->Id() << '\n';
      std::cout << "MElement Node IDs: " << (MasterElement().Nodes()[0])->Id() << " "
                << (MasterElement().Nodes()[1])->Id() << '\n';
      std::cout << "SPROJXI_0: " << sprojxi[0] << " SPROJXI_1: " << sprojxi[1] << '\n';
      std::cout << "MPROJXI_0: " << mprojxi[0] << " MPROJXI_1: " << mprojxi[1] << '\n';
      FOUR_C_THROW("IntegrateOverlap: Significant overlap ignored S%i M%i!", SlaveElement().Id(),
          MasterElement().Id());
    }
  }

  else if (!s0hasproj && s1hasproj && !m0hasproj && !m1hasproj)
  {
    if ((-1.0 + MORTARPROJTOL <= sprojxi[1]) && (sprojxi[1] <= 1.0 - MORTARPROJTOL))
    {
      std::cout << "SElement Node IDs: " << (SlaveElement().Nodes()[0])->Id() << " "
                << (SlaveElement().Nodes()[1])->Id() << '\n';
      std::cout << "MElement Node IDs: " << (MasterElement().Nodes()[0])->Id() << " "
                << (MasterElement().Nodes()[1])->Id() << '\n';
      std::cout << "SPROJXI_0: " << sprojxi[0] << " SPROJXI_1: " << sprojxi[1] << '\n';
      std::cout << "MPROJXI_0: " << mprojxi[0] << " MPROJXI_1: " << mprojxi[1] << '\n';
      FOUR_C_THROW("IntegrateOverlap: Significant overlap ignored S%i M%i!", SlaveElement().Id(),
          MasterElement().Id());
    }
  }

  else if (!s0hasproj && !s1hasproj && m0hasproj && !m1hasproj)
  {
    if ((-1.0 + MORTARPROJTOL <= mprojxi[0]) && (mprojxi[0] <= 1.0 - MORTARPROJTOL))
    {
      std::cout << "SElement Node IDs: " << (SlaveElement().Nodes()[0])->Id() << " "
                << (SlaveElement().Nodes()[1])->Id() << '\n';
      std::cout << "MElement Node IDs: " << (MasterElement().Nodes()[0])->Id() << " "
                << (MasterElement().Nodes()[1])->Id() << '\n';
      std::cout << "SPROJXI_0: " << sprojxi[0] << " SPROJXI_1: " << sprojxi[1] << '\n';
      std::cout << "MPROJXI_0: " << mprojxi[0] << " MPROJXI_1: " << mprojxi[1] << '\n';
      FOUR_C_THROW("IntegrateOverlap: Significant overlap ignored S%i M%i!", SlaveElement().Id(),
          MasterElement().Id());
    }
  }

  else if (!s0hasproj && !s1hasproj && !m0hasproj && m1hasproj)
  {
    if ((-1.0 + MORTARPROJTOL <= mprojxi[1]) && (mprojxi[1] <= 1.0 - MORTARPROJTOL))
    {
      std::cout << "SElement Node IDs: " << (SlaveElement().Nodes()[0])->Id() << " "
                << (SlaveElement().Nodes()[1])->Id() << '\n';
      std::cout << "MElement Node IDs: " << (MasterElement().Nodes()[0])->Id() << " "
                << (MasterElement().Nodes()[1])->Id() << '\n';
      std::cout << "SPROJXI_0: " << sprojxi[0] << " SPROJXI_1: " << sprojxi[1] << '\n';
      std::cout << "MPROJXI_0: " << mprojxi[0] << " MPROJXI_1: " << mprojxi[1] << '\n';
      FOUR_C_THROW("IntegrateOverlap: Significant overlap ignored S%i M%i!", SlaveElement().Id(),
          MasterElement().Id());
    }
  }

  /* CASE 6 (OVERLAP):
   feasible projection found for all 4 outer element nodes
   (this can happen due to the necessary projection tolerance!!!)     */

  else if (s0hasproj && s1hasproj && m0hasproj && m1hasproj)
  {
    overlap = true;

    // switch, since nurbs might not be ordered anti-clockwise!!!
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() > 0.)
    {
      // internal case 1 for global CASE 6
      // (equivalent to global CASE 7, slave fully projects onto master)
      if ((sprojxi[0] < 1.0) && (sprojxi[1] > -1.0))
      {
        sxia = -1.0;
        sxib = 1.0;
        mxia = sprojxi[1];  // local node numbering always anti-clockwise!!!
        mxib = sprojxi[0];
        // std::cout << "Problem solved with internal case 1!" << std::endl;
      }

      // internal case 2 for global CASE 6
      // (equivalent to global CASE 8, master fully projects onto slave)
      else if ((mprojxi[0] < 1.0) && (mprojxi[1] > -1.0))
      {
        mxia = -1.0;
        mxib = 1.0;
        sxia = mprojxi[1];  // local node numbering always anti-clockwise!!!
        sxib = mprojxi[0];
        // std::cout << "Problem solved with internal case 2!" << std::endl;
      }

      // internal case 3 for global CASE 6
      // (equivalent to global CASE 9, both nodes no. 0 project successfully)
      else if ((sprojxi[0] < 1.0 + MORTARPROJLIM) && (mprojxi[0] < 1.0 + MORTARPROJLIM))
      {
        sxia = -1.0;
        sxib = mprojxi[0];  // local node numbering always anti-clockwise!!!
        mxia = -1.0;
        mxib = sprojxi[0];
        // std::cout << "Problem solved with internal case 3!" << std::endl;
      }

      // internal case 4 for global CASE 6
      // (equivalent to global CASE 10, both nodes no. 1 project successfully)
      else if ((sprojxi[1] > -1.0 - MORTARPROJLIM) && (mprojxi[1] > -1.0 - MORTARPROJLIM))
      {
        sxia = mprojxi[1];
        sxib = 1.0;  // local node numbering always anti-clockwise!!!
        mxia = sprojxi[1];
        mxib = 1.0;
        // std::cout << "Problem solved with internal case 4!" << std::endl;
      }

      // unknown internal case for global CASE 6
      else
      {
        std::cout << "MORTAR::Coupling2d::DetectOverlap " << '\n'
                  << "has detected '4 projections'-case for Sl./Ma. pair " << SlaveElement().Id()
                  << "/" << MasterElement().Id() << '\n';
        std::cout << "SElement Node IDs: " << (SlaveElement().Nodes()[0])->Id() << " "
                  << (SlaveElement().Nodes()[1])->Id() << '\n';
        std::cout << "MElement Node IDs: " << (MasterElement().Nodes()[0])->Id() << " "
                  << (MasterElement().Nodes()[1])->Id() << '\n';
        std::cout << "SPROJXI_0: " << sprojxi[0] << " SPROJXI_1: " << sprojxi[1] << '\n';
        std::cout << "MPROJXI_0: " << mprojxi[0] << " MPROJXI_1: " << mprojxi[1] << '\n';
        FOUR_C_THROW("DetectOverlap: Unknown overlap case found in global case 6!");
      }
    }

    else
    {
      // fully projecting slave element: equivalent to global case 7
      if ((sprojxi[0] > -1.) && (sprojxi[1] < 1.))
      {
        sxia = -1.0;
        sxib = 1.0;
        mxia = sprojxi[0];
        mxib = sprojxi[1];
      }
      // fully projecting master element: equivalent to global case 8
      else if ((mprojxi[0] > -1.) && (mprojxi[1] < 1.))
      {
        mxia = -1.0;
        mxib = 1.0;
        sxia = mprojxi[0];
        sxib = mprojxi[1];
      }
      // equivalent to global case 15
      else if ((sprojxi[1] < 1.0 + MORTARPROJLIM) && (mprojxi[0] < 1.0 + MORTARPROJLIM))
      {
        sxia = mprojxi[0];
        sxib = 1.;
        mxia = -1.;
        mxib = sprojxi[1];
      }
      // equivalent to global case 16
      else if ((sprojxi[0] > -1.0 - MORTARPROJLIM) && (mprojxi[1] > -1.0 - MORTARPROJLIM))
      {
        sxia = -1.;
        sxib = mprojxi[1];
        mxia = sprojxi[0];
        mxib = 1.;
      }
      else
      {
        std::cout << "MORTAR::Coupling2d::DetectOverlap " << '\n'
                  << "has detected '4 projections'-case for Sl./Ma. pair " << SlaveElement().Id()
                  << "/" << MasterElement().Id() << '\n';
        std::cout << "SElement Node IDs: " << (SlaveElement().Nodes()[0])->Id() << " "
                  << (SlaveElement().Nodes()[1])->Id() << '\n';
        std::cout << "MElement Node IDs: " << (MasterElement().Nodes()[0])->Id() << " "
                  << (MasterElement().Nodes()[1])->Id() << '\n';
        std::cout << "SPROJXI_0: " << sprojxi[0] << " SPROJXI_1: " << sprojxi[1] << '\n';
        std::cout << "MPROJXI_0: " << mprojxi[0] << " MPROJXI_1: " << mprojxi[1] << '\n';
        FOUR_C_THROW("DetectOverlap: Unknown overlap case found in global case 6!");
      }
    }
  }

  /* CASES 7-8 (OVERLAP):
   feasible projections found for both nodes of one element, this
   means one of the two elements is projecting fully onto the other!  */

  else if (s0hasproj && s1hasproj && !m0hasproj && !m1hasproj)
  {
    overlap = true;
    sxia = -1.0;
    sxib = 1.0;
    // nurbs may not be numbered anti-clockwise
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() > 0.)
    {
      mxia = sprojxi[1];  // local node numbering always anti-clockwise!!!
      mxib = sprojxi[0];
    }
    else
    {
      mxia = sprojxi[0];
      mxib = sprojxi[1];
    }
  }

  else if (!s0hasproj && !s1hasproj && m0hasproj && m1hasproj)
  {
    overlap = true;
    mxia = -1.0;
    mxib = 1.0;
    // nurbs may not be numbered anti-clockwise
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() > 0.)
    {
      sxia = mprojxi[1];  // local node numbering always anti-clockwise!!!
      sxib = mprojxi[0];
    }
    else
    {
      sxia = mprojxi[0];
      sxib = mprojxi[1];
    }
  }

  /* CASES 9-10 (OVERLAP):
   feasible projections found for one node of each element, due to
   node numbering only identical local node ID pairs possible!        */

  else if (s0hasproj && !s1hasproj && m0hasproj && !m1hasproj)
  {
    // do the two elements really have an overlap?
    if ((sprojxi[0] > -1.0 + MORTARPROJLIM) && (mprojxi[0] > -1.0 + MORTARPROJLIM))
    {
      overlap = true;
      sxia = -1.0;
      sxib = mprojxi[0];  // local node numbering always anti-clockwise!!!
      mxia = -1.0;
      mxib = sprojxi[0];
    }
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() < 0.)
    {
      std::cout << "SElement: " << SlaveElement().NodeIds()[0] << " " << SlaveElement().NodeIds()[1]
                << '\n';
      std::cout << "MElement: " << MasterElement().NodeIds()[0] << " "
                << MasterElement().NodeIds()[1] << '\n';
      std::cout << "s0: " << s0hasproj << " s1: " << s1hasproj << '\n';
      std::cout << "m0: " << m0hasproj << " m1: " << m1hasproj << '\n';
      FOUR_C_THROW("IntegrateOverlap: Unknown overlap case found!");
    }
  }

  else if (!s0hasproj && s1hasproj && !m0hasproj && m1hasproj)
  {
    // do the two elements really have an overlap?
    if ((sprojxi[1] < 1.0 - MORTARPROJLIM) && (mprojxi[1] < 1.0 - MORTARPROJLIM))
    {
      overlap = true;
      sxia = mprojxi[1];
      sxib = 1.0;  // local node numbering always anti-clockwise!!!
      mxia = sprojxi[1];
      mxib = 1.0;
    }
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() < 0.)
    {
      std::cout << "SElement: " << SlaveElement().NodeIds()[0] << " " << SlaveElement().NodeIds()[1]
                << '\n';
      std::cout << "MElement: " << MasterElement().NodeIds()[0] << " "
                << MasterElement().NodeIds()[1] << '\n';
      std::cout << "s0: " << s0hasproj << " s1: " << s1hasproj << '\n';
      std::cout << "m0: " << m0hasproj << " m1: " << m1hasproj << '\n';
      FOUR_C_THROW("IntegrateOverlap: Unknown overlap case found!");
    }
  }

  /* CASES 11-14 (OVERLAP):
   feasible projections found for 3 out of the total 4 nodes,
   this can either lead to cases 7/8 or 9/10!                         */
  else if (s0hasproj && s1hasproj && m0hasproj && !m1hasproj)
  {
    overlap = true;

    // switch, since nurbs might not be ordered anti-clockwise!!!
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() > 0.)
    {
      // equivalent to global case 7
      if (mprojxi[0] > 1.0)
      {
        sxia = -1.0;
        sxib = 1.0;
        mxia = sprojxi[1];  // local node numbering always anti-clockwise!!!
        mxib = sprojxi[0];
      }
      // equivalent to global case 9
      else
      {
        sxia = -1.0;
        sxib = mprojxi[0];  // local node numbering always anti-clockwise!!!
        mxia = -1.0;
        mxib = sprojxi[0];
      }
    }
    else
    {
      // equivalent to global case 7
      if (mprojxi[0] < -1.)
      {
        sxia = -1.0;
        sxib = 1.0;
        mxia = sprojxi[0];
        mxib = sprojxi[1];
      }
      // equivalent to global case 15
      else
      {
        sxia = mprojxi[0];
        sxib = 1.;
        mxia = -1.;
        mxib = sprojxi[1];
      }
    }
  }

  else if (s0hasproj && s1hasproj && !m0hasproj && m1hasproj)
  {
    overlap = true;

    // switch, since nurbs might not be ordered anti-clockwise!!!
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() > 0.)
    {
      // equivalent to global case 7
      if (mprojxi[1] < -1.0)
      {
        sxia = -1.0;
        sxib = 1.0;
        mxia = sprojxi[1];  // local node numbering always anti-clockwise!!!
        mxib = sprojxi[0];
      }
      // equivalent to global case 10
      else
      {
        sxia = mprojxi[1];
        sxib = 1.0;  // local node numbering always anti-clockwise!!!
        mxia = sprojxi[1];
        mxib = 1.0;
      }
    }

    else
    {
      // equivalent to global case 7
      if (mprojxi[1] > 1.)
      {
        sxia = -1.;
        sxib = 1.;
        mxia = sprojxi[0];
        mxib = sprojxi[1];
      }
      // equivalent to global case 16
      else
      {
        sxia = -1.;
        sxib = mprojxi[1];
        mxia = sprojxi[0];
        mxib = 1.;
      }
    }
  }

  else if (s0hasproj && !s1hasproj && m0hasproj && m1hasproj)
  {
    overlap = true;

    // switch, since nurbs might not be ordered anti-clockwise!!!
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() > 0.)
    {
      // equivalent to global case 8
      if (sprojxi[0] > 1.0)
      {
        mxia = -1.0;
        mxib = 1.0;
        sxia = mprojxi[1];  // local node numbering always anti-clockwise!!!
        sxib = mprojxi[0];
      }
      // equivalent to global case 9
      else
      {
        sxia = -1.0;
        sxib = mprojxi[0];  // local node numbering always anti-clockwise!!!
        mxia = -1.0;
        mxib = sprojxi[0];
      }
    }
    else
    {
      // equivalent to global case 8
      if (sprojxi[0] < -1.)
      {
        mxia = -1.0;
        mxib = 1.0;
        sxia = mprojxi[0];
        sxib = mprojxi[1];
      }
      // equivalent to global case 16
      else
      {
        sxia = -1.;
        sxib = mprojxi[1];
        mxia = sprojxi[0];
        mxib = 1.;
      }
    }
  }

  else if (!s0hasproj && s1hasproj && m0hasproj && m1hasproj)
  {
    overlap = true;

    // switch, since nurbs might not be ordered anti-clockwise!!!
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() > 0.)
    {
      // equivalent to global case 8
      if (sprojxi[1] < -1.0)
      {
        mxia = -1.0;
        mxib = 1.0;
        sxia = mprojxi[1];  // local node numbering always anti-clockwise!!!
        sxib = mprojxi[0];
      }
      // equivalent to global case 10
      else
      {
        sxia = mprojxi[1];
        sxib = 1.0;  // local node numbering always anti-clockwise!!!
        mxia = sprojxi[1];
        mxib = 1.0;
      }
    }
    else
    {
      // equivalent to global case 8
      if (sprojxi[1] > 1.)
      {
        mxia = -1.;
        mxib = 1.;
        sxia = mprojxi[0];
        sxib = mprojxi[1];
      }
      // equivalent to global case 15
      else
      {
        sxia = mprojxi[0];
        sxib = 1.;
        mxia = -1.;
        mxib = sprojxi[1];
      }
    }
  }

  /* CASES 15-16 (OVERLAP):
   feasible projections found for one node of each element, due to
   node numbering opposite local node ID pairs possible only for nurbs! */

  else if (!s0hasproj && s1hasproj && m0hasproj && !m1hasproj)
  {
    // only possible, if slave and master side have opposite normal-fac
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() > 0.)
    {
      std::cout << "SElement: " << SlaveElement().NodeIds()[0] << " " << SlaveElement().NodeIds()[1]
                << '\n';
      std::cout << "MElement: " << MasterElement().NodeIds()[0] << " "
                << MasterElement().NodeIds()[1] << '\n';
      std::cout << "s0: " << s0hasproj << " s1: " << s1hasproj << '\n';
      std::cout << "m0: " << m0hasproj << " m1: " << m1hasproj << '\n';
      FOUR_C_THROW("IntegrateOverlap: Unknown overlap case found!");
    }
    if (sprojxi[0] < -1. || mprojxi[0] > 1.)
      overlap = false;
    else
    {
      overlap = true;
      sxia = mprojxi[0];
      sxib = 1.;
      mxia = -1.;
      mxib = sprojxi[1];
    }
  }

  else if (s0hasproj && !s1hasproj && !m0hasproj && m1hasproj)
  {
    // only possible, if slave and master side have opposite normal-fac
    if (SlaveElement().NormalFac() * MasterElement().NormalFac() > 0.)
    {
      std::cout << "SElement: " << SlaveElement().NodeIds()[0] << " " << SlaveElement().NodeIds()[1]
                << '\n';
      std::cout << "MElement: " << MasterElement().NodeIds()[0] << " "
                << MasterElement().NodeIds()[1] << '\n';
      std::cout << "s0: " << s0hasproj << " s1: " << s1hasproj << '\n';
      std::cout << "m0: " << m0hasproj << " m1: " << m1hasproj << '\n';
      FOUR_C_THROW("IntegrateOverlap: Unknown overlap case found!");
    }
    if (sprojxi[0] > 1.)
      overlap = false;
    else
    {
      overlap = true;
      sxia = -1.;
      sxib = mprojxi[1];
      mxia = sprojxi[0];
      mxib = 1.;
    }
  }

  /* CASE DEFAULT: unknown overlap case                                  */
  else
  {
    std::cout << "SElement: " << SlaveElement().NodeIds()[0] << " " << SlaveElement().NodeIds()[1]
              << '\n';
    std::cout << "MElement: " << MasterElement().NodeIds()[0] << " " << MasterElement().NodeIds()[1]
              << '\n';
    std::cout << "s0: " << s0hasproj << " s1: " << s1hasproj << '\n';
    std::cout << "m0: " << m0hasproj << " m1: " << m1hasproj << '\n';
    FOUR_C_THROW("IntegrateOverlap: Unknown overlap case found!");
  }

  // check for 1:1 node projections and for infeasible limits
  if ((sxia < -1.0) || (sxib > 1.0) || (mxia < -1.0) || (mxib > 1.0))
  {
    if (abs(sxia + 1.0) < MORTARPROJLIM) sxia = -1.0;
    if (abs(sxib - 1.0) < MORTARPROJLIM) sxib = 1.0;
    if (abs(mxia + 1.0) < MORTARPROJLIM) mxia = -1.0;
    if (abs(mxib - 1.0) < MORTARPROJLIM) mxib = 1.0;

    if ((sxia < -1.0) || (sxib > 1.0) || (mxia < -1.0) || (mxib > 1.0))
    {
      //      std::cout << "Slave: " << sxia << " " << sxib << std::endl;
      //      std::cout << "Master: " << mxia << " " << mxib << std::endl;
      //      FOUR_C_THROW("IntegrateOverlap: Determined infeasible limits!");
      std::cout << "WARNING: IntegrateOverlap: Determined infeasible limits!" << '\n';
      overlap = false;
    }
  }

  // update integration limits in xiproj_
  xiproj_[0] = sxia;
  xiproj_[1] = sxib;
  xiproj_[2] = mxia;
  xiproj_[3] = mxib;

  // store overlap information
  overlap_ = overlap;

  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool MORTAR::Coupling2d::IntegrateOverlap(const Teuchos::RCP<MORTAR::ParamsInterface>& mparams_ptr)
{
  // explicitly defined shape function type needed
  if (ShapeFcn() == INPAR::MORTAR::shape_undefined)
    FOUR_C_THROW("IntegrateOverlap called without specific shape function defined!");

  /**********************************************************************/
  /* INTEGRATION                                                        */
  /* Depending on overlap and the xiproj_ entries integrate the Mortar  */
  /* matrices D and M on the overlap of the current sl / ma pair.       */
  /**********************************************************************/

  // no integration if no overlap
  if (!overlap_) return false;

  // set segmentation status of all slave nodes
  // (hassegment_ of a slave node is true if ANY segment/cell
  // is integrated that contributes to this slave node)
  int nnodes = SlaveElement().num_node();
  DRT::Node** mynodes = SlaveElement().Nodes();
  if (!mynodes) FOUR_C_THROW("Null pointer!");
  for (int k = 0; k < nnodes; ++k)
  {
    auto* mycnode = dynamic_cast<MORTAR::Node*>(mynodes[k]);
    if (!mycnode) FOUR_C_THROW("Null pointer!");
    mycnode->HasSegment() = true;
  }

  // local working copies of input variables
  double sxia = xiproj_[0];
  double sxib = xiproj_[1];
  double mxia = xiproj_[2];
  double mxib = xiproj_[3];

  // *******************************************************************
  // different options for mortar integration
  // *******************************************************************
  // (1) no quadratic element(s) involved -> linear LM interpolation
  // (2) quadratic element(s) involved -> quadratic LM interpolation
  // (3) quadratic element(s) involved -> linear LM interpolation
  // (4) quadratic element(s) involved -> piecew. linear LM interpolation
  // *******************************************************************
  INPAR::MORTAR::LagMultQuad lmtype = LagMultQuad();

  // *******************************************************************
  // cases (1), (2) and (3)
  // *******************************************************************
  if (!Quad() || (Quad() && lmtype == INPAR::MORTAR::lagmult_quad) ||
      (Quad() && lmtype == INPAR::MORTAR::lagmult_lin) ||
      (Quad() && lmtype == INPAR::MORTAR::lagmult_const))
  {
    // do the overlap integration (integrate and linearize both M and gap)
    MORTAR::Integrator::Impl(SlaveElement(), MasterElement(), interface_params())
        ->IntegrateSegment2D(SlaveElement(), sxia, sxib, MasterElement(), mxia, mxib, Comm());
  }

  // *******************************************************************
  // case (4)
  // *******************************************************************
  else if (Quad() && lmtype == INPAR::MORTAR::lagmult_pwlin)
  {
    FOUR_C_THROW("Piecewise linear LM interpolation not (yet?) implemented in 2D");
  }

  // *******************************************************************
  // undefined case
  // *******************************************************************
  else if (Quad() && lmtype == INPAR::MORTAR::lagmult_undefined)
  {
    FOUR_C_THROW(
        "Lagrange multiplier interpolation for quadratic elements undefined\n"
        "If you are using 2nd order mortar elements, you need to specify LM_QUAD in MORTAR "
        "COUPLING section");
  }

  // *******************************************************************
  // other cases
  // *******************************************************************
  else
  {
    FOUR_C_THROW("IntegrateOverlap: Invalid case for 2D mortar coupling LM interpolation");
  }

  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MORTAR::Coupling2dManager::Coupling2dManager(DRT::Discretization& idiscret, int dim, bool quad,
    Teuchos::ParameterList& params, MORTAR::Element* sele, std::vector<MORTAR::Element*> mele)
    : idiscret_(idiscret),
      dim_(dim),
      quad_(quad),
      imortar_(params),
      sele_(sele),
      mele_(std::move(mele))
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MORTAR::Coupling2dManager::IntegrateCoupling(
    const Teuchos::RCP<MORTAR::ParamsInterface>& mparams_ptr)
{
  // decide which type of numerical integration scheme

  //**********************************************************************
  // STANDARD INTEGRATION (SEGMENTS)
  //**********************************************************************
  if (IntType() == INPAR::MORTAR::inttype_segments)
  {
    // loop over all master elements associated with this slave element
    for (int m = 0; m < (int)MasterElements().size(); ++m)
    {
      // create Coupling2d object and push back
      Coupling().push_back(Teuchos::rcp(
          new Coupling2d(idiscret_, dim_, quad_, imortar_, SlaveElement(), MasterElement(m))));

      // project the element pair
      Coupling()[m]->Project();

      // check for element overlap
      Coupling()[m]->DetectOverlap();
    }

    // special treatment of boundary elements
    // calculate consistent dual shape functions for this element
    consist_dual_shape();

    // do mortar integration
    for (int m = 0; m < (int)MasterElements().size(); ++m)
      Coupling()[m]->IntegrateOverlap(mparams_ptr);
  }

  //**********************************************************************
  // FAST INTEGRATION (ELEMENTS)
  //**********************************************************************
  else if (IntType() == INPAR::MORTAR::inttype_elements ||
           IntType() == INPAR::MORTAR::inttype_elements_BS)
  {
    if ((int)MasterElements().size() == 0) return;

    // bool for boundary segmentation
    bool boundary_ele = false;

    // *******************************************************************
    // different options for mortar integration
    // *******************************************************************
    // (1) no quadratic element(s) involved -> linear LM interpolation
    // (2) quadratic element(s) involved -> quadratic LM interpolation
    // (3) quadratic element(s) involved -> linear LM interpolation
    // (4) quadratic element(s) involved -> piecew. linear LM interpolation
    // *******************************************************************
    INPAR::MORTAR::LagMultQuad lmtype = LagMultQuad();

    // *******************************************************************
    // cases (1), (2) and (3)
    // *******************************************************************
    if (!Quad() || (Quad() && lmtype == INPAR::MORTAR::lagmult_quad) ||
        (Quad() && lmtype == INPAR::MORTAR::lagmult_lin))
    {
      MORTAR::Integrator::Impl(SlaveElement(), MasterElement(0), imortar_)
          ->IntegrateEleBased2D(SlaveElement(), MasterElements(), &boundary_ele, idiscret_.Comm());

      // Perform Boundary Segmentation if required
      if (IntType() == INPAR::MORTAR::inttype_elements_BS)
      {
        if (boundary_ele)
        {
          // std::cout << "Boundary segmentation for element: " << SlaveElement().Id() << "\n" ;
          // switch, if consistent boundary modification chosen
          if (CORE::UTILS::IntegralValue<INPAR::MORTAR::ConsistentDualType>(
                  imortar_, "LM_DUAL_CONSISTENT") != INPAR::MORTAR::consistent_none &&
              ShapeFcn() != INPAR::MORTAR::shape_standard  // so for petrov-Galerkin and dual
          )
          {
            // loop over all master elements associated with this slave element
            for (int m = 0; m < (int)MasterElements().size(); ++m)
            {
              // create Coupling2d object and push back
              Coupling().push_back(Teuchos::rcp(new Coupling2d(
                  idiscret_, dim_, quad_, imortar_, SlaveElement(), MasterElement(m))));

              // project the element pair
              Coupling()[m]->Project();

              // check for element overlap
              Coupling()[m]->DetectOverlap();
            }

            // calculate consistent dual shape functions for this element
            consist_dual_shape();

            // do mortar integration
            for (int m = 0; m < (int)MasterElements().size(); ++m)
              Coupling()[m]->IntegrateOverlap(mparams_ptr);
          }

          else
          {
            for (int m = 0; m < (int)MasterElements().size(); ++m)
            {
              // create Coupling2d object and push back
              Coupling().push_back(Teuchos::rcp(new Coupling2d(
                  idiscret_, dim_, quad_, imortar_, SlaveElement(), MasterElement(m))));

              // project the element pair
              Coupling()[m]->Project();

              // check for element overlap
              Coupling()[m]->DetectOverlap();

              // integrate the element overlap
              Coupling()[m]->IntegrateOverlap(mparams_ptr);
            }
          }
        }
        else
        {
          // nothing
        }
      }
      else
      {
        // nothing
      }
    }
  }
  //**********************************************************************
  // INVALID
  //**********************************************************************
  else
  {
    FOUR_C_THROW("Invalid type of numerical integration");
  }

  // free memory of consistent dual shape function coefficient matrix
  SlaveElement().MoData().ResetDualShape();
  SlaveElement().MoData().ResetDerivDualShape();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool MORTAR::Coupling2dManager::EvaluateCoupling(
    const Teuchos::RCP<MORTAR::ParamsInterface>& mparams_ptr)
{
  if (MasterElements().size() == 0) return false;

  // decide which type of coupling should be evaluated
  auto algo = CORE::UTILS::IntegralValue<INPAR::MORTAR::AlgorithmType>(imortar_, "ALGORITHM");

  //*********************************
  // Mortar Contact
  //*********************************
  if (algo == INPAR::MORTAR::algorithm_mortar or algo == INPAR::MORTAR::algorithm_gpts)
  {
    IntegrateCoupling(mparams_ptr);
  }

  //*********************************
  // Error
  //*********************************
  else
    FOUR_C_THROW("chose contact algorithm not supported!");

  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MORTAR::Coupling2dManager::consist_dual_shape()
{
  // For standard shape functions no modification is necessary
  // A switch erlier in the process improves computational efficiency
  auto consistent =
      CORE::UTILS::IntegralValue<INPAR::MORTAR::ConsistentDualType>(imortar_, "LM_DUAL_CONSISTENT");
  if (ShapeFcn() == INPAR::MORTAR::shape_standard || consistent == INPAR::MORTAR::consistent_none)
    return;

  // Consistent modification not yet checked for constant LM interpolation
  if (Quad() && LagMultQuad() == INPAR::MORTAR::lagmult_const &&
      consistent != INPAR::MORTAR::consistent_none)
    FOUR_C_THROW(
        "ERROR: Consistent dual shape functions not yet checked for constant LM interpolation!");

  // not implemented for nurbs yet
  if (SlaveElement().Shape() == CORE::FE::CellType::nurbs3)
    FOUR_C_THROW("Consistent dual shape functions not yet implmented for nurbs");

  // do nothing if there are no coupling pairs
  if (Coupling().size() == 0) return;

  // detect entire overlap
  double ximin = 1.0;
  double ximax = -1.0;
  std::map<int, double> dximin;
  std::map<int, double> dximax;
  std::vector<std::map<int, double>> ximaps(4);

  // loop over all master elements associated with this slave element
  for (const auto& coupling : Coupling())
  {
    // go on, if this s/m pair has no overlap
    if (not coupling->Overlap()) continue;

    double sxia = coupling->XiProj()[0];
    double sxib = coupling->XiProj()[1];

    // get element contact integration area
    // and for contact derivatives of beginning and end
    if (sxia < ximin) ximin = sxia;
    if (sxib > ximax) ximax = sxib;
  }

  // no overlap: the applied dual shape functions don't matter, as the integration domain is void
  if ((ximax == -1.0 && ximin == 1.0) || (ximax - ximin < 4. * MORTARINTLIM)) return;

  // fully projecting element: no modification necessary
  if (ximin == -1. && ximax == 1.) return;

  // calculate consistent dual schape functions (see e.g. Cichosz et.al.:
  // Consistent treatment of boundaries with mortar contact formulations, CMAME 2010

  // get number of nodes of present slave element
  int nnodes = SlaveElement().num_node();

  // compute entries to bi-ortho matrices me/de with Gauss quadrature
  MORTAR::ElementIntegrator integrator(SlaveElement().Shape());

  // prepare for calculation of dual shape functions
  CORE::LINALG::SerialDenseMatrix me(nnodes, nnodes, true);
  CORE::LINALG::SerialDenseMatrix de(nnodes, nnodes, true);

  for (int gp = 0; gp < integrator.nGP(); ++gp)
  {
    CORE::LINALG::SerialDenseVector sval(nnodes);
    CORE::LINALG::SerialDenseMatrix sderiv(nnodes, 1, true);

    // coordinates and weight
    std::array<double, 2> eta = {integrator.Coordinate(gp, 0), 0.0};
    double wgt = integrator.Weight(gp);

    // coordinate transformation sxi->eta (slave MORTAR::Element->Overlap)
    std::array<double, 2> sxi = {0.0, 0.0};
    sxi[0] = 0.5 * (1.0 - eta[0]) * ximin + 0.5 * (1.0 + eta[0]) * ximax;

    // evaluate trace space shape functions
    if (LagMultQuad() == INPAR::MORTAR::lagmult_lin)
    {
      SlaveElement().evaluate_shape_lag_mult_lin(
          INPAR::MORTAR::shape_standard, sxi.data(), sval, sderiv, nnodes);
    }
    else
      SlaveElement().EvaluateShape(sxi.data(), sval, sderiv, nnodes);

    // evaluate the two slave side Jacobians
    double dxdsxi = SlaveElement().Jacobian(sxi.data());
    double dsxideta = -0.5 * ximin + 0.5 * ximax;

    // integrate dual shape matrices de, me and their linearizations
    for (int j = 0; j < nnodes; ++j)
    {
      // de and linearization
      de(j, j) += wgt * sval[j] * dxdsxi * dsxideta;

      // me and linearization
      for (int k = 0; k < nnodes; ++k)
      {
        me(j, k) += wgt * sval[j] * sval[k] * dxdsxi * dsxideta;
      }
    }
  }

  // declare dual shape functions coefficient matrix
  CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes, true);

  // compute matrix A_e for linear interpolation of quadratic element
  if (LagMultQuad() == INPAR::MORTAR::lagmult_lin)
  {
    // how many non-zero nodes
    const int nnodeslin = 2;

    // reduce me to non-zero nodes before inverting
    CORE::LINALG::Matrix<nnodeslin, nnodeslin> melin;
    for (int j = 0; j < nnodeslin; ++j)
      for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

    // invert bi-ortho matrix melin
    CORE::LINALG::Inverse(melin);

    // re-inflate inverse of melin to full size
    CORE::LINALG::SerialDenseMatrix invme(nnodes, nnodes, true);
    for (int j = 0; j < nnodeslin; ++j)
      for (int k = 0; k < nnodeslin; ++k) invme(j, k) = melin(j, k);

    // get solution matrix with dual parameters
    CORE::LINALG::multiply(ae, de, invme);
  }
  // compute matrix A_e for all other cases
  else
    CORE::LINALG::InvertAndMultiplyByCholesky(me, de, ae);

  // store ae matrix in slave element data container
  SlaveElement().MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
}

FOUR_C_NAMESPACE_CLOSE