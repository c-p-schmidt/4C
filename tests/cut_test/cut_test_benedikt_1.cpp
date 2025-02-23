// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_meshintersection.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_tetmeshintersection.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cut_test_utils.hpp"

void test_benedikt1()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.05030222279960472;
    tri3_xyze(1, 0) = 0.5164896689179502;
    tri3_xyze(2, 0) = 0.0;
    tri3_xyze(0, 1) = 0.05022204783542078;
    tri3_xyze(1, 1) = 0.5141348415851575;
    tri3_xyze(2, 1) = 0.0;
    tri3_xyze(0, 2) = 0.05026213531751275;
    tri3_xyze(1, 2) = 0.5153122552515541;
    tri3_xyze(2, 2) = -0.0025;
    nids.clear();
    nids.push_back(1781);
    nids.push_back(1784);
    nids.push_back(1786);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.05022204783542078;
    tri3_xyze(1, 0) = 0.5141348415851575;
    tri3_xyze(2, 0) = 0.0;
    tri3_xyze(0, 1) = 0.05022204783542078;
    tri3_xyze(1, 1) = 0.5141348415851579;
    tri3_xyze(2, 1) = -0.005;
    tri3_xyze(0, 2) = 0.05026213531751275;
    tri3_xyze(1, 2) = 0.5153122552515541;
    tri3_xyze(2, 2) = -0.0025;
    nids.clear();
    nids.push_back(1784);
    nids.push_back(1785);
    nids.push_back(1786);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.05022204783542078;
    tri3_xyze(1, 0) = 0.5141348415851579;
    tri3_xyze(2, 0) = -0.005;
    tri3_xyze(0, 1) = 0.05030222279960472;
    tri3_xyze(1, 1) = 0.5164896689179506;
    tri3_xyze(2, 1) = -0.005;
    tri3_xyze(0, 2) = 0.05026213531751275;
    tri3_xyze(1, 2) = 0.5153122552515541;
    tri3_xyze(2, 2) = -0.0025;
    nids.clear();
    nids.push_back(1785);
    nids.push_back(1782);
    nids.push_back(1786);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }

  sidecount = 595;
  {
    Core::LinAlg::SerialDenseMatrix quad4_xyze(3, 4);

    quad4_xyze(0, 0) = 0.05022204783542078;
    quad4_xyze(1, 0) = 0.5141348415851575;
    quad4_xyze(2, 0) = 0.0;
    quad4_xyze(0, 1) = 0.05015420376099927;
    quad4_xyze(1, 1) = 0.5117796267385427;
    quad4_xyze(2, 1) = 0.0;
    quad4_xyze(0, 2) = 0.05015420376099927;
    quad4_xyze(1, 2) = 0.511779626738543;
    quad4_xyze(2, 2) = -0.005;
    quad4_xyze(0, 3) = 0.05022204783542078;
    quad4_xyze(1, 3) = 0.5141348415851579;
    quad4_xyze(2, 3) = -0.005;

    nids.clear();
    nids.push_back(1784);
    nids.push_back(1787);
    nids.push_back(1788);
    nids.push_back(1785);

    intersection.add_cut_side(595, nids, quad4_xyze, Core::FE::CellType::quad4);
  }


  Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

  hex8_xyze(0, 0) = 0.049219426993576;
  hex8_xyze(1, 0) = 0.5159099025766973;
  hex8_xyze(2, 0) = -0.00375;

  hex8_xyze(0, 1) = 0.04745166004060965;
  hex8_xyze(1, 1) = 0.514142135623731;
  hex8_xyze(2, 1) = -0.00375;

  hex8_xyze(0, 2) = 0.049219426993576;
  hex8_xyze(1, 2) = 0.5123743686707646;
  hex8_xyze(2, 2) = -0.00375;

  hex8_xyze(0, 3) = 0.05098719394654241;
  hex8_xyze(1, 3) = 0.5141421356237309;
  hex8_xyze(2, 3) = -0.00375;

  hex8_xyze(0, 4) = 0.04921942699357601;
  hex8_xyze(1, 4) = 0.5159099025766973;
  hex8_xyze(2, 4) = -0.00125;

  hex8_xyze(0, 5) = 0.04745166004060967;
  hex8_xyze(1, 5) = 0.5141421356237309;
  hex8_xyze(2, 5) = -0.00125;

  hex8_xyze(0, 6) = 0.04921942699357602;
  hex8_xyze(1, 6) = 0.5123743686707646;
  hex8_xyze(2, 6) = -0.00125;

  hex8_xyze(0, 7) = 0.05098719394654238;
  hex8_xyze(1, 7) = 0.5141421356237309;
  hex8_xyze(2, 7) = -0.00125;

  {
    int data[] = {-360197818, 1068053321, 2078800790, 1071678037, -343597384, -1083262895};
    std::memcpy(&hex8_xyze(0, 0), data, 3 * sizeof(double));
  }
  {
    int data[] = {-2146969310, 1067994005, 893385748, 1071674330, -343597384, -1083262895};
    std::memcpy(&hex8_xyze(0, 1), data, 3 * sizeof(double));
  }
  {
    int data[] = {-360197817, 1068053321, -292029295, 1071670622, -343597384, -1083262895};
    std::memcpy(&hex8_xyze(0, 2), data, 3 * sizeof(double));
  }
  {
    int data[] = {1426573684, 1068112638, 893385747, 1071674330, -343597384, -1083262895};
    std::memcpy(&hex8_xyze(0, 3), data, 3 * sizeof(double));
  }
  {
    int data[] = {-360197816, 1068053321, 2078800790, 1071678037, 1202590843, -1084982559};
    std::memcpy(&hex8_xyze(0, 4), data, 3 * sizeof(double));
  }
  {
    int data[] = {-2146969307, 1067994005, 893385747, 1071674330, 1202590843, -1084982559};
    std::memcpy(&hex8_xyze(0, 5), data, 3 * sizeof(double));
  }
  {
    int data[] = {-360197814, 1068053321, -292029295, 1071670622, 1202590843, -1084982559};
    std::memcpy(&hex8_xyze(0, 6), data, 3 * sizeof(double));
  }
  {
    int data[] = {1426573679, 1068112638, 893385747, 1071674330, 1202590843, -1084982559};
    std::memcpy(&hex8_xyze(0, 7), data, 3 * sizeof(double));
  }



  nids.clear();
  for (int i = 0; i < 8; ++i) nids.push_back(i);

  int eid = 192551;

  intersection.add_element(eid, nids, hex8_xyze, Core::FE::CellType::hex8);
  intersection.cut_test_cut(true, Cut::VCellGaussPts_DirectDivergence);
}
