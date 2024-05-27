/*----------------------------------------------------------------------------*/
/*! \file
  *
 *  \brief class representing a geometrical side
 *

\level 2

 *------------------------------------------------------------------------------------------------*/

#ifndef FOUR_C_CUT_SIDE_HPP
#define FOUR_C_CUT_SIDE_HPP

#include "4C_config.hpp"

#include "4C_cut_edge.hpp"
#include "4C_cut_element.hpp"
#include "4C_cut_tolerance.hpp"

#include <Shards_BasicTopologies.hpp>
#include <Shards_CellTopologyTraits.hpp>

FOUR_C_NAMESPACE_OPEN

namespace CORE::GEO
{
  namespace CUT
  {
    class Facet;
    class LineSegment;
    class BoundaryCell;
    class LineSegmentList;
    class Cycle;
    class BoundingBox;

    enum MarkedActions
    {
      mark_and_do_nothing = -1,
      mark_and_create_boundarycells = 0
    };

    /*! \brief Base class for dealing sides in cut algorithm
     *
     *  The maximal allowed dimension of sides is 2! */
    class Side
    {
     public:
      /** create a new level set side
       *
       *  Create a new level set side using the current problem dimension
       *
       *  \param sid (in) : side id
       *
       *  \author hiermeier
       *  \date 08/16 */
      static Side* CreateLevelSetSide(const int& sid);

      /// Create a concrete side (using dis-type for identification)
      static Side* Create(const CORE::FE::CellType& sidetype, const int& sid,
          const std::vector<Node*>& nodes, const std::vector<Edge*>& edges);

      /// Create a concrete side (using shards-key for identification)
      static Side* Create(const unsigned& shardskey, const int& sid,
          const std::vector<Node*>& nodes, const std::vector<Edge*>& edges);

      /** \brief constructor
       *
       *  side  (in) : side id
       *  nodes (in) : nodes of this side
       *  edges (in) : edges of this side */
      Side(int sid, const std::vector<Node*>& nodes, const std::vector<Edge*>& edges);

      /// destructor
      virtual ~Side() = default;
      /// \brief Returns the ID of the side
      int Id() const { return sid_; }

      /// returns the geometric shape of this side
      virtual CORE::FE::CellType Shape() const = 0;

      /// element dimension
      virtual unsigned Dim() const = 0;

      /// problem dimension
      virtual unsigned ProbDim() const = 0;

      /// number of nodes
      virtual unsigned NumNodes() const = 0;

      /// \brief Returns the topology data for the side from Shards library
      virtual const CellTopologyData* Topology() const = 0;

      /// \brief Set the side ID to the input value
      void SetId(int sid) { sid_ = sid; }

      /// \brief Set the side ID to the input value
      void set_marked_side_properties(int markedid, enum MarkedActions markedaction)
      {
        // No combination of cut-sides and marked sides!!
        if (sid_ > -1) FOUR_C_THROW("Currently a marked side and a cut side can not co-exist");
        // Number of markings on the surface should not be more than 1
        if (markedsidemap_.size() > 0)
          FOUR_C_THROW("Currently more than one mark on a side is NOT possible.");

        markedsidemap_.insert(std::pair<enum MarkedActions, int>(
            CORE::GEO::CUT::mark_and_create_boundarycells, markedid));
      }

      std::map<enum MarkedActions, int>& GetMarkedsidemap() { return markedsidemap_; }

      /// \brief Returns true if this is a cut side
      bool IsCutSide() { return (sid_ > -1); }

      /// \brief Returns true if this is a cut side
      bool IsBoundaryCellSide() { return (IsCutSide() or is_marked_background_side()); }

      /// \brief Returns true if this is a marked side
      bool is_marked_background_side() { return (!IsCutSide() and (markedsidemap_.size() > 0)); }

      /// \brief Register this element in which the side is a part of
      void Register(Element* element) { elements_.insert(element); }

      /*void CollectElements( plain_element_set & elements )
       {
       std::copy( elements_.begin(), elements_.end(), std::inserter( elements, elements.begin() ) );
       }*/

      /*! \brief is this side closer to the start-point as the other side?
       *
       *  Set is_closer and return TRUE if check was successful. */
      template <class T>
      bool IsCloserSide(const T& startpoint_xyz, CORE::GEO::CUT::Side* other, bool& is_closer)
      {
        if (startpoint_xyz.M() != ProbDim())
          FOUR_C_THROW("The dimension of startpoint_xyz is wrong! (probdim = %d)", ProbDim());
        return IsCloserSide(startpoint_xyz.A(), other, is_closer);
      }

      /*! \brief Calculates the points at which the side is cut by this edge */
      virtual bool Cut(Mesh& mesh, Edge& edge, PointSet& cut_points) = 0;

      virtual bool JustParallelCut(Mesh& mesh, Edge& edge, PointSet& cut_points, int idx = -1)
      {
        FOUR_C_THROW("JustParallelCut is not implemented for the levelset!");
        return false;
      }

      /*! \brief get all edges adjacent to given local coordinates */
      template <class T1>
      void EdgeAt(const T1& rs, std::vector<Edge*>& edges)
      {
        if (static_cast<unsigned>(rs.M()) != Dim())
          FOUR_C_THROW("The dimension of rs is wrong! (dim = %d)", Dim());

        EdgeAt(rs.A(), edges);
      }

      /*! \brief get the global coordinates on side at given local coordinates */
      template <class T1, class T2>
      void PointAt(const T1& rs, T2& xyz)
      {
        if (static_cast<unsigned>(xyz.M()) < ProbDim())
          FOUR_C_THROW("The dimension of xyz is wrong! (probdim = %d)", ProbDim());
        if (static_cast<unsigned>(rs.M()) != Dim())
          FOUR_C_THROW("The dimension of rs is wrong! (dim = %d)", Dim());

        PointAt(rs.A(), xyz.A());

        // fill remaining entries with zeros
        std::fill(xyz.A() + ProbDim(), xyz.A() + xyz.M(), 0.0);
      }

      /*! \brief get global coordinates of the center of the side */
      template <class T1>
      void SideCenter(T1& midpoint)
      {
        if (midpoint.M() != ProbDim())
          FOUR_C_THROW("The dimension of xyz is wrong! (probdim = %d)", ProbDim());
        SideCenter(midpoint.A());
      }

      /*! \brief Calculates the local coordinates (\c rsd) with respect to the element shape
       *  from its global coordinates (xyz), return \c TRUE if successful. The last coordinate
       *  of \c rsd is the distance of the n-dimensional point \c xyz to the embedded
       *  side. */
      template <class T1, class T2>
      bool LocalCoordinates(
          const T1& xyz, T2& rsd, bool allow_dist = false, double tol = POSITIONTOL)
      {
        if (static_cast<unsigned>(xyz.M()) < ProbDim())
          FOUR_C_THROW("The dimension of xyz is wrong! (probdim = %d)", ProbDim());
        if (static_cast<unsigned>(rsd.M()) < ProbDim())
          FOUR_C_THROW("The dimension of rsd is wrong! (dim = %d)", Dim());

        const bool check = LocalCoordinates(xyz.A(), rsd.A(), allow_dist, tol);

        std::fill(rsd.A() + ProbDim(), rsd.A() + rsd.M(), 0.0);

        return check;
      }

      /*! \brief get local coordinates (rst) with respect to the element shape
       * for all the corner points */
      virtual void local_corner_coordinates(double* rst_corners) = 0;

      /*! \brief lies point with given coordinates within this side? */
      template <class T1, class T2>
      bool WithinSide(const T1& xyz, T2& rs, double& dist)
      {
        if (static_cast<unsigned>(xyz.M()) != ProbDim())
          FOUR_C_THROW("The dimension of xyz is wrong! (probdim = %d)", ProbDim());
        if (static_cast<unsigned>(rs.M()) != Dim())
          FOUR_C_THROW("The dimension of rs is wrong! (dim = %d)", Dim());

        return WithinSide(xyz.A(), rs.A(), dist);
      }

      /* \brief compute the cut of a ray through two points with the 2D space defined by
       * the side */
      template <class T1, class T2>
      bool RayCut(const T1& p1_xyz, const T1& p2_xyz, T2& rs, double& line_xi)
      {
        if (p1_xyz.M() != ProbDim())
          FOUR_C_THROW("The dimension of xyz is wrong! (probdim = %d)", ProbDim());
        if (rs.M() != Dim()) FOUR_C_THROW("The dimension of rs is wrong! (dim = %d)", Dim());
        return RayCut(p1_xyz.A(), p2_xyz.A(), rs.A(), line_xi);
      }

      /*! \brief Calculates the normal vector with respect to the element shape
       *  at local coordinates \c rs */
      template <class T1, class T2>
      void Normal(const T1& rs, T2& normal, bool unitnormal = true)
      {
        if (static_cast<unsigned>(normal.M()) != ProbDim())
          FOUR_C_THROW("The dimension of xyz is wrong! (probdim = %d)", ProbDim());
        if (static_cast<unsigned>(rs.M()) != Dim())
          FOUR_C_THROW("The dimension of rs is wrong! (dim = %d)", Dim());

        Normal(rs.A(), normal.A(), unitnormal);
      }

      /* \brief Calculates a Basis of two tangential vectors (non-orthogonal!) and
       * the normal vector with respect to the element shape at local coordinates rs,
       * basis vectors have norm=1. */
      template <class T1, class T2>
      void Basis(const T1& rs, T2& t1, T2& t2, T2& n)
      {
        if (static_cast<unsigned>(t1.M()) != ProbDim())
          FOUR_C_THROW("The dimension of xyz is wrong! (probdim = %d)", ProbDim());
        if (static_cast<unsigned>(rs.M()) != Dim())
          FOUR_C_THROW("The dimension of rs is wrong! (dim = %d)", Dim());

        Basis(rs.A(), t1.A(), t2.A(), n.A());
      }

      /* \brief Calculates a Basis of two tangential vectors (non-orthogonal!) and
       * the normal vector with respect to the element shape at local coordinates rs,
       * basis vectors have norm=1 */
      template <class T>
      void BasisAtCenter(T& t1, T& t2, T& n)
      {
        if (static_cast<unsigned>(t1.M()) != ProbDim())
          FOUR_C_THROW("The dimension of xyz is wrong! (probdim = %d)", ProbDim());

        BasisAtCenter(t1.A(), t2.A(), n.A());
      }

      /*! \brief Returns the global coordinates of the nodes of this side */
      virtual void Coordinates(double* xyze) const = 0;

     private:
      /*! \brief Fixes the matrix shape and returns the global coordinates of the
       *  nodes of this side
       *
       *  Be aware that this routine is more expensive than the calling function
       *  because we have to copy the data in the end. So if it's possible to give
       *  the matrix with the correct shape, always do it!
       *
       *  \author hiermeier \date 01/17 */
      template <class T>
      void fix_shape_and_get_coordinates(T& xyze) const
      {
        CORE::LINALG::SerialDenseMatrix xyze_corrected(ProbDim(), NumNodes());

        Coordinates(xyze_corrected.values());

        // copy the result back into the given matrix
        for (unsigned c = 0; c < NumNodes(); ++c)
        {
          CORE::LINALG::SerialDenseMatrix xyz(
              Teuchos::View, &xyze_corrected(0, c), ProbDim(), ProbDim(), 1);
          std::copy(xyz.values(), xyz.values() + ProbDim(), &xyze(0, c));
          // fill the rows out of range with zeros
          std::fill(&xyze(0, c) + ProbDim(), &xyze(0, c) + xyze.numRows(), 0.0);
        }

        // fill columns out of range with zeros
        for (unsigned c = NumNodes(); c < static_cast<unsigned>(xyze.numCols()); ++c)
          std::fill(&xyze(0, c), &xyze(0, c) + xyze.numRows(), 0.0);
      }

     public:
      template <class T>
      void Coordinates(T& xyze) const
      {
        if (static_cast<unsigned>(xyze.numRows()) < ProbDim())
          FOUR_C_THROW("The row dimension of xyze is wrong! (probdim = %d)", ProbDim());
        if (static_cast<unsigned>(xyze.numCols()) < NumNodes())
          FOUR_C_THROW("The col dimension of xyze is wrong! (numNodesSide = %d)", NumNodes());

        // if the matrix
        if (static_cast<unsigned>(xyze.numRows()) > ProbDim() or
            static_cast<unsigned>(xyze.numCols()) > NumNodes())
        {
          fix_shape_and_get_coordinates(xyze);
        }
        else
        {
          Coordinates(xyze.values());
        }
      }

      /*! \brief Returns true if this is a cut side */
      virtual bool IsLevelSetSide() { return false; };

      /** create facets on the background sides of the element
       *
       *  For all these facets, parent side is an element side */
      virtual void MakeOwnedSideFacets(Mesh& mesh, Element* element, plain_facet_set& facets);

      /** \brief create facets on the cut sides of the element
       *
       *  For all these facets, parent side is a cut side.
       *
       *  \note See also the derived LevelSet version for information.  */
      virtual void MakeInternalFacets(Mesh& mesh, Element* element, plain_facet_set& facets);

      void MakeInternalFacets(
          Mesh& mesh, Element* element, const Cycle& points, plain_facet_set& facets);

      virtual bool IsCut();

      //   virtual bool DoTriangulation() { return true; }

      // bool FullSideCut() { return cut_lines_.size()==edges_.size() and facets_.size()==1; }

      bool OnSide(const PointSet& points);

      bool OnEdge(Point* point);

      bool OnEdge(Line* line);

      /// All points of this side are in the other side (based on the Id())
      bool AllPointsCommon(Side& side);

      bool HaveCommonNode(Side& side);

      // Finds if this sides is touched by the other side at the point "p"
      bool IsTouched(Side& other, Point* p);

      // Finds if this side is topologically touched by the other side in the point "p"
      // Purely based on the location of the point might not even be in the node of other side
      bool IsTouchedAt(Side* other, Point* p);

      bool HaveCommonEdge(Side& side);

      Element* CommonElement(Side* other);

      //   virtual void ExchangeFacetSide( Side * side, bool cutsurface ) = 0;

      void AddPoint(Point* cut_point);

      void AddLine(Line* cut_line);

      Facet* FindFacet(const std::vector<Point*>& facet_points);

      /*! \brief Find Cut Lines for two Cut Sides, which have more than two common cut points!
       *  (This happens if the cutsides are in the same plane !) */
      bool find_touching_cut_lines(Mesh& mesh, Element* element, Side& side, const PointSet& cut);

      /*! \brief Find Cut Lines for two Cut Sides specially based on a discretization,
       *  which have more than two common cut points!
       *
       *  (This happens if the cutsides are in the same plane or due to numerical tolerances! */
      virtual bool find_ambiguous_cut_lines(
          Mesh& mesh, Element* element, Side& side, const PointSet& cut);

      // Find part of this side (cut points on the cut_edges), that lies on the other side
      // (parallel)
      virtual bool find_parallel_intersection(
          Mesh& mesh, Element* element, Side& side, const PointSet& cut, point_line_set& new_lines);


      // create paralle cut surface between two sides
      virtual bool create_parallel_cut_surface(Mesh& mesh, Element* element, Side& other,
          const PointSet& cut, std::vector<Point*>* cut_point_for_lines_out = nullptr);

      void GetBoundaryCells(plain_boundarycell_set& bcells);

      void Print();

      template <class T>
      Node* OnNode(const T& x)
      {
        if (x.M() != ProbDim())
          FOUR_C_THROW("x has the wrong dimension! (probDim = %d)", ProbDim());

        T nx;
        for (std::vector<Node*>::iterator i = nodes_.begin(); i != nodes_.end(); ++i)
        {
          Node* n = *i;
          n->Coordinates(nx.A());
          nx.Update(-1, x, 1);
          if (nx.Norm2() <= (x.NormInf() * POSITIONTOL + n->point()->Tolerance()))
          {
            return n;
          }
        }
        return nullptr;
      }

      const std::vector<Edge*>& Edges() const { return edges_; }

      /*!
       brief Returns the edge of this side with given begin and end points
       */
      Edge* FindEdge(Point* begin, Point* end);

      const std::vector<Node*>& Nodes() const { return nodes_; }

      virtual bool find_cut_points_dispatch(Mesh& mesh, Element* element, Side& side, Edge& e);

      /*!
       \brief Calculate the points at which the other side intersects with this considered side
       */
      virtual bool find_cut_points(Mesh& mesh, Element* element, Side& other);

      /*!
       \brief Draw cut lines between the cut points of this edge and "other"
       */
      bool find_cut_lines(Mesh& mesh, Element* element, Side& other);

      /*!
       \brief Get (not calculate) all the cut points between this side and "other"
       */
      void GetCutPoints(Element* element, Side& other, PointSet& cuts);

      /*!
       \brief Get all the cut points that are produced by this edge
       */
      const PointSet& CutPoints() const { return cut_points_; }

      const std::vector<Line*>& CutLines() const { return cut_lines_; }

      const plain_element_set& Elements() const { return elements_; }

      const std::vector<Facet*>& Facets() const { return facets_; }

      /// returns true if the hole is inside the facet
      bool HoleOfFacet(Facet& facet, const std::vector<Cycle>& hole);

      /// Gets a cutting side of this cutside
      void GetCuttingSide(Side* cuttingside) { cutting_sides_.insert(cuttingside); }

      /// Gets selfcutpoints of this cutside
      void GetSelfCutPoints(PointSet& selfcutpoints)
      {
        for (PointSet::iterator i = selfcutpoints.begin(); i != selfcutpoints.end(); ++i)
        {
          Point* selfcutpoint = *i;
          self_cut_points_.insert(selfcutpoint);
        }
      }

      /// Gets a selfcutnode of this cutside
      void GetSelfCutNode(Node* selfcutnode) { self_cut_nodes_.insert(selfcutnode); }

      /// Gets a selfcutedge of this cutside
      void GetSelfCutEdge(Edge* selfcutedge) { self_cut_edges_.insert(selfcutedge); }

      /// Gets a selfcuttriangle of this cutside
      void GetSelfCutTriangle(std::vector<CORE::GEO::CUT::Point*> selfcuttriangle)
      {
        self_cut_triangles_.push_back(selfcuttriangle);
      }

      /// Gets the selfcutposition of this cutside and spreads the positional information
      void GetSelfCutPosition(Point::PointPosition p);

      /// Changes the selfcutposition of this cutside and spreads the positional information
      void change_self_cut_position(Point::PointPosition p);

      /// Erase a cuttingside from this cutside because the bounding box found too much
      void EraseCuttingSide(Side* nocuttingside) { cutting_sides_.erase(nocuttingside); }

      /// Returns all cutting sides of this cutside
      const plain_side_set& CuttingSides() const { return cutting_sides_; }

      /// Returns all selfcutpoints of this cutside
      const PointSet& SelfCutPoints() const { return self_cut_points_; }

      /// Returns all selfcutnodes of this cutside
      const plain_node_set& SelfCutNodes() const { return self_cut_nodes_; }

      /// Returns all selfcutedges of this cutside
      const plain_edge_set& SelfCutEdges() const { return self_cut_edges_; }

      /// Returns all selfcuttriangles of this cutside
      const std::vector<std::vector<CORE::GEO::CUT::Point*>>& SelfCutTriangles() const
      {
        return self_cut_triangles_;
      }

      /// Returns the selfcutposition of this cutside
      Point::PointPosition SelfCutPosition() { return selfcutposition_; }

      /// replace "nod" associated with this side by the new node "replwith"
      void replaceNodes(Node* nod, Node* replwith);

      /// get bounding volume
      const BoundingBox& GetBoundingVolume() const { return *boundingvolume_; }

      /// Remove point p from this side
      void RemovePoint(Point* p) { cut_points_.erase(p); };

      /// Parallel Cut Surfaces on this side (represented by the points)
      std::set<std::set<Point*>>& get_parallel_cut_surfaces() { return parallel_cut_surfaces_; }


     protected:
      bool AllOnNodes(const PointSet& points);

      /** @name All these functions have to be implemented in the derived classes!
       *
       *  \remark Please note, that none of these functions has any inherent
       *          dimension checks! Be careful if you access them directly. Each of
       *          these functions has a public alternative which checks the dimensions
       *          and is therefore much safer.                      hiermeier 07/16 */
      /// @{

      /*! \brief is this side closer to the starting-point as the other side?
       *
       *  Set is_closer and return TRUE if check was successful. */
      virtual bool IsCloserSide(
          const double* startpoint_xyz, CORE::GEO::CUT::Side* other, bool& is_closer) = 0;

      /*! \brief get all edges adjacent to given local coordinates */
      virtual void EdgeAt(const double* rs, std::vector<Edge*>& edges) = 0;

      /*! \brief get the global coordinates on side at given local coordinates */
      virtual void PointAt(const double* rs, double* xyz) = 0;

      /*! \brief get global coordinates of the center of the side */
      virtual void SideCenter(double* midpoint) = 0;

      /*! \brief Calculates the local coordinates (rsd) with respect to the element shape
       *  from its global coordinates (xyz), return TRUE if successful. The last coordinate
       *  of \c rsd is the distance of the n-dimensional point \c xyz to the embedded
       *  side. */
      virtual bool LocalCoordinates(
          const double* xyz, double* rsd, bool allow_dist, double tol) = 0;

      /*! \brief Does the point with given coordinates lie within this side? */
      virtual bool WithinSide(const double* xyz, double* rs, double& dist) = 0;

      /* \brief compute the cut of a ray through two points with the 2D space defined by the side */
      virtual bool RayCut(
          const double* p1_xyz, const double* p2_xyz, double* rs, double& line_xi) = 0;

      /*! \brief Calculates the normal vector with respect to the element shape at local coordinates
       * rs */
      virtual void Normal(const double* rs, double* normal, bool unitnormal) = 0;

      /* \brief Calculates a Basis of two tangential vectors (non-orthogonal!) and
       * the normal vector with respect to the element shape at local coordinates rs,
       * basis vectors have norm=1. */
      virtual void Basis(const double* rs, double* t1, double* t2, double* n) = 0;

      /* \brief Calculates a Basis of two tangential vectors (non-orthogonal!) and
       * the normal vector with respect to the element shape at local coordinates rs,
       * basis vectors have norm=1 */
      virtual void BasisAtCenter(double* t1, double* t2, double* n) = 0;

      /// @}

     private:
      /** \brief Get the default uncut facet number per side
       *
       *  The number is dependent on the dimension of the parent element.
       *
       *  \author hiermeier \date 01/17 */
      unsigned uncut_facet_number_per_side() const;

      /// Simplifies topological connection in the case, when side contains two parallel cut
      /// surfaces, which are self-intersecting.
      void simplify_mixed_parallel_cut_surface(Mesh& mesh, Element* element, Side& other,
          std::set<Point*>& new_surface, std::vector<Point*>& cut_points_for_lines);

      /// Does this Side have another parallel cut surface than this?
      bool has_mixed_parallel_cut_surface(const std::set<Point*>& surface)
      {
        return (parallel_cut_surfaces_.size() > 0 &&
                parallel_cut_surfaces_.find(surface) == parallel_cut_surfaces_.end());
      }

     private:
      int sid_;

      // Marked side additions:
      std::map<CORE::GEO::CUT::MarkedActions, int> markedsidemap_;
      // -----------------------

      std::vector<Node*> nodes_;

      std::vector<Edge*> edges_;

      plain_element_set elements_;

      std::vector<Line*> cut_lines_;

      PointSet cut_points_;

      std::vector<Facet*> facets_;

      /// in some extreme cases surface can be partially
      /// parallel to the other side
      std::set<std::set<Point*>> parallel_cut_surfaces_;

      /// all sides which are cutting this cutside
      plain_side_set cutting_sides_;

      /// all selfcutpoints of this cutside
      PointSet self_cut_points_;

      /// all selfcutnodes of this cutside
      plain_node_set self_cut_nodes_;

      /// all selfcutedges of this cutside
      plain_edge_set self_cut_edges_;

      /// all selfcuttriangles of this cutside
      std::vector<std::vector<CORE::GEO::CUT::Point*>> self_cut_triangles_;

      /// the selfcutposition of this cutside shows if it is inside or outside the other structure
      /// body
      Point::PointPosition selfcutposition_;

      /// the bounding volume of the side
      Teuchos::RCP<BoundingBox> boundingvolume_;

    };  // class side

    /*! \brief Implementation of the concrete side element
     *
     *  The class is a template on the problem dimension \c probdim,
     *                             the side type \c sidetype
     *                             the number of nodes per side \c numNodesSide
     *                             the side dimension \c dim
     *
     *  \author hiermeier */
    template <unsigned probdim, CORE::FE::CellType sidetype,
        unsigned numNodesSide = CORE::FE::num_nodes<sidetype>,
        unsigned dim = CORE::FE::dim<sidetype>>
    class ConcreteSide : public Side, public ConcreteElement<probdim, sidetype>
    {
     public:
      /// constructor
      ConcreteSide(int sid, const std::vector<Node*>& nodes, const std::vector<Edge*>& edges)
          : Side(sid, nodes, edges),
            ConcreteElement<probdim, sidetype>(-1, std::vector<Side*>(0), nodes, false)
      {
        // sanity check
        if (dim > 2)
          FOUR_C_THROW(
              "The element dimension of sides is not allowed to be greater"
              " than 2!");
      }

      /// Returns the geometrical shape of this side
      CORE::FE::CellType Shape() const override { return sidetype; }

      /// element dimension
      unsigned Dim() const override { return dim; }

      /// problem dimension
      unsigned ProbDim() const override { return probdim; }

      unsigned NumNodes() const override { return numNodesSide; }

      /// Returns the topology data for the side from Shards library
      const CellTopologyData* Topology() const override
      {
        switch (sidetype)
        {
          case CORE::FE::CellType::tri3:
            return shards::getCellTopologyData<shards::Triangle<3>>();
            break;
          case CORE::FE::CellType::quad4:
            return shards::getCellTopologyData<shards::Quadrilateral<4>>();
            break;
          case CORE::FE::CellType::line2:
            return shards::getCellTopologyData<shards::Line<2>>();
            break;
          default:
            FOUR_C_THROW("Unknown sidetype! (%d | %s)\n", sidetype,
                CORE::FE::CellTypeToString(sidetype).c_str());
            break;
        }
        exit(EXIT_FAILURE);
      }

      /// Calculates the points at which the side is cut by this edge
      bool Cut(Mesh& mesh, Edge& edge, PointSet& cut_points) override
      {
        return edge.Cut(mesh, *this, cut_points);
      }

      bool JustParallelCut(Mesh& mesh, Edge& edge, PointSet& cut_points, int idx = -1) override
      {
        return edge.JustParallelCut(mesh, *this, cut_points, idx);
      }

      /** \brief Is this side closer to the start-point as the other side?
       *
       *  check based on ray-tracing technique set is_closer and return
       *  \TRUE if check was successful */
      bool IsCloserSide(const CORE::LINALG::Matrix<probdim, 1>& startpoint_xyz,
          CORE::GEO::CUT::Side* other, bool& is_closer);

      /// get all edges adjacent to given local coordinates
      void EdgeAt(const CORE::LINALG::Matrix<dim, 1>& rs, std::vector<Edge*>& edges)
      {
        switch (sidetype)
        {
          case CORE::FE::CellType::tri3:
          {
            if (fabs(rs(1)) < REFERENCETOL) edges.push_back(Edges()[0]);
            if (fabs(rs(0) + rs(1) - 1) < REFERENCETOL) edges.push_back(Edges()[1]);
            if (fabs(rs(0)) < REFERENCETOL) edges.push_back(Edges()[2]);
            break;
          }
          case CORE::FE::CellType::quad4:
          {
            if (fabs(rs(1) + 1) < REFERENCETOL) edges.push_back(Edges()[0]);
            if (fabs(rs(0) - 1) < REFERENCETOL) edges.push_back(Edges()[1]);
            if (fabs(rs(1) - 1) < REFERENCETOL) edges.push_back(Edges()[2]);
            if (fabs(rs(0) + 1) < REFERENCETOL) edges.push_back(Edges()[3]);
            break;
          }
          case CORE::FE::CellType::line2:
          {
            FOUR_C_THROW("If we need this, the edges will degenerate to nodes!");
            break;
          }
          default:
          {
            throw "Unknown/unsupported side type! \n";
            break;
          }
        }  // switch (sidetype)
      }

      /** \brief get the global coordinates on side at given local coordinates
       *
       *  \param rs  (in)  : parameter space coordinates
       *  \param xyz (out) : corresponding spatial coordinates */
      void PointAt(const CORE::LINALG::Matrix<dim, 1>& rs, CORE::LINALG::Matrix<probdim, 1>& xyz)
      {
        ConcreteElement<probdim, sidetype>::PointAt(rs, xyz);
      }

      /** \brief get global coordinates of the center of the side
       *
       *  \param midpoint (out) : mid-point spatial coordinates */
      void SideCenter(CORE::LINALG::Matrix<probdim, 1>& midpoint)
      {
        ConcreteElement<probdim, sidetype>::ElementCenter(midpoint);
      }

      ///  lies point with given coordinates within this side?
      bool WithinSide(const CORE::LINALG::Matrix<probdim, 1>& xyz, CORE::LINALG::Matrix<dim, 1>& rs,
          double& dist);

      /// compute the cut of a ray through two points with the 2D space defined by the side
      bool RayCut(const CORE::LINALG::Matrix<probdim, 1>& p1_xyz,
          const CORE::LINALG::Matrix<probdim, 1>& p2_xyz, CORE::LINALG::Matrix<dim, 1>& rs,
          double& line_xi);

      /** \brief Calculates the local coordinates (rst) with respect to the element shape from its
       *  global coordinates (xyz), return TRUE if successful
       *
       *  \remark The last coordinate of the variable rsd holds the distance to the side. */
      bool LocalCoordinates(const CORE::LINALG::Matrix<probdim, 1>& xyz,
          CORE::LINALG::Matrix<probdim, 1>& rsd, bool allow_dist = false, double tol = POSITIONTOL);

      /// get local coordinates (rst) with respect to the element shape for all the corner points
      void local_corner_coordinates(double* rst_corners) override
      {
        switch (sidetype)
        {
          case CORE::FE::CellType::tri3:
          {
            const double rs[6] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
            std::copy(rs, rs + 6, rst_corners);
            break;
          }
          case CORE::FE::CellType::quad4:
          {
            const double rs[8] = {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0};
            std::copy(rs, rs + 8, rst_corners);
            break;
          }
          case CORE::FE::CellType::line2:
          {
            const double r[2] = {-1.0, 1.0};
            std::copy(r, r + 2, rst_corners);
            break;
          }
          default:
          {
            FOUR_C_THROW("Unknown/unsupported side type!");
            exit(EXIT_FAILURE);
          }
        }
      }

      /// Calculates the normal vector with respect to the element shape at local coordinates xsi
      void Normal(const CORE::LINALG::Matrix<dim, 1>& xsi, CORE::LINALG::Matrix<probdim, 1>& normal,
          bool unitnormal = true)
      {
        // get derivatives at pos
        CORE::LINALG::Matrix<probdim, numNodesSide> side_xyze(true);
        this->Coordinates(side_xyze);

        CORE::LINALG::Matrix<dim, numNodesSide> deriv(true);
        CORE::LINALG::Matrix<dim, probdim> A(true);

        CORE::FE::shape_function_deriv1<sidetype>(xsi, deriv);
        A.MultiplyNT(deriv, side_xyze);

        switch (dim)
        {
          // 1-dimensional side-element embedded in 2-dimensional space
          case 1:
          {
            if (probdim == 3)
              FOUR_C_THROW(
                  "Dimension mismatch. A 1-dimensional element in a 3-dimensional space is "
                  "not a side, but a edge!");

            normal(0) = A(0, 1);   //   dy/dxi
            normal(1) = -A(0, 0);  // - dx/dxi

            break;
          }
          // 2-dimensional side element embedded in 3-dimensional space
          case 2:
          {
            // cross product to get the normal at the point
            normal(0) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
            normal(1) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
            normal(2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

            break;
          }
          default:
            FOUR_C_THROW("Unsupported element dimension!");
            break;
        }
        // force unit length
        if (unitnormal)
        {
          double norm = normal.Norm2();
          normal.Scale(1. / norm);
        }
      }

      /** Calculates a Basis of two tangential vectors (non-orthogonal!) and
       *  the normal vector with respect to the element shape at center of the side.
       *  All basis vectors are of unit length */
      void BasisAtCenter(CORE::LINALG::Matrix<probdim, 1>& t1, CORE::LINALG::Matrix<probdim, 1>& t2,
          CORE::LINALG::Matrix<probdim, 1>& n)
      {
        CORE::LINALG::Matrix<dim, 1> center_rs(CORE::FE::getLocalCenterPosition<dim>(sidetype));
        Basis(center_rs, t1, t2, n);
      }

      /** Calculates a Basis of two tangential vectors (non-orthogonal!) and
       * the normal vector with respect to the element shape at local coordinates xsi.
       * All basis vectors are of unit length. */
      void Basis(const CORE::LINALG::Matrix<dim, 1>& xsi, CORE::LINALG::Matrix<probdim, 1>& t1,
          CORE::LINALG::Matrix<probdim, 1>& t2, CORE::LINALG::Matrix<probdim, 1>& n)
      {
        // get derivatives at pos
        CORE::LINALG::Matrix<probdim, numNodesSide> side_xyze(true);
        this->Coordinates(side_xyze);

        CORE::LINALG::Matrix<dim, numNodesSide> deriv(true);
        CORE::LINALG::Matrix<dim, probdim> A(true);

        CORE::FE::shape_function_deriv1<sidetype>(xsi, deriv);
        A.MultiplyNT(deriv, side_xyze);

        // set the first tangential vector
        t1(0) = A(0, 0);
        t1(1) = A(0, 1);
        t1(2) = A(0, 2);

        t1.Scale(1. / t1.Norm2());


        switch (dim)
        {
          case 1:
          {
            // there is no 2nd tangential vector in the 1-D case
            t2 = 0;
            break;
          }
          case 2:
          {
            // set the second tangential vector
            t2(0) = A(1, 0);
            t2(1) = A(1, 1);
            t2(2) = A(1, 2);

            t2.Scale(1. / t2.Norm2());
            break;
          }
        }
        // calculate the normal
        Normal(xsi, n);
      }

      /// get coordinates of side
      void Coordinates(CORE::LINALG::Matrix<probdim, numNodesSide>& xyze_surfaceElement) const
      {
        Coordinates(xyze_surfaceElement.A());
      }

      /*! \brief Returns the global coordinates of the nodes of this side */
      void Coordinates(double* xyze) const override
      {
        ConcreteElement<probdim, sidetype>::Coordinates(xyze);
      }

     protected:
      /// derived
      bool IsCloserSide(
          const double* startpoint_xyz, CORE::GEO::CUT::Side* other, bool& is_closer) override
      {
        const CORE::LINALG::Matrix<probdim, 1> startpoint_xyz_mat(startpoint_xyz, true);
        return IsCloserSide(startpoint_xyz_mat, other, is_closer);
      }

      /// derived
      void EdgeAt(const double* rs, std::vector<Edge*>& edges) override
      {
        const CORE::LINALG::Matrix<dim, 1> rs_mat(rs, true);  // create view
        EdgeAt(rs_mat, edges);
      }

      /// derived
      void PointAt(const double* rs, double* xyz) override
      {
        const CORE::LINALG::Matrix<dim, 1> rs_mat(rs, true);  // create view
        CORE::LINALG::Matrix<probdim, 1> xyz_mat(xyz, true);  // create view
        PointAt(rs_mat, xyz_mat);
      }

      /// derived
      void SideCenter(double* midpoint) override
      {
        CORE::LINALG::Matrix<probdim, 1> midpoint_mat(midpoint, true);  // create view
        SideCenter(midpoint_mat);
      }

      /// derived
      bool LocalCoordinates(const double* xyz, double* rsd, bool allow_dist = false,
          double tol = POSITIONTOL) override
      {
        const CORE::LINALG::Matrix<probdim, 1> xyz_mat(xyz, true);  // create view
        CORE::LINALG::Matrix<probdim, 1> rsd_mat(rsd, true);        // create view
        return LocalCoordinates(xyz_mat, rsd_mat, allow_dist, tol);
      }

      /// derived
      bool WithinSide(const double* xyz, double* rs, double& dist) override
      {
        const CORE::LINALG::Matrix<probdim, 1> xyz_mat(xyz, true);  // create view
        CORE::LINALG::Matrix<dim, 1> rs_mat(rs, true);              // create view
        return WithinSide(xyz_mat, rs_mat, dist);
      }

      /// derived
      bool RayCut(const double* p1_xyz, const double* p2_xyz, double* rs, double& line_xi) override
      {
        const CORE::LINALG::Matrix<probdim, 1> p1_xyz_mat(p1_xyz, true);  // create view
        const CORE::LINALG::Matrix<probdim, 1> p2_xyz_mat(p2_xyz, true);  // create view
        CORE::LINALG::Matrix<dim, 1> rs_mat(rs, true);                    // create view
        return RayCut(p1_xyz_mat, p2_xyz_mat, rs_mat, line_xi);
      }

      /// derived
      void Normal(const double* rs, double* normal, bool unitnormal = true) override
      {
        const CORE::LINALG::Matrix<dim, 1> rs_mat(rs, true);        // create view
        CORE::LINALG::Matrix<probdim, 1> normal_mat(normal, true);  // create view
        Normal(rs_mat, normal_mat, unitnormal);
      }

      /// derived
      void Basis(const double* rs, double* t1, double* t2, double* n) override
      {
        const CORE::LINALG::Matrix<dim, 1> rs_mat(rs, true);  // create view
        CORE::LINALG::Matrix<probdim, 1> t1_mat(t1, true);    // create view
        CORE::LINALG::Matrix<probdim, 1> t2_mat(t2, true);    // create view
        CORE::LINALG::Matrix<probdim, 1> n_mat(n, true);      // create view
        Basis(rs_mat, t1_mat, t2_mat, n_mat);
      }

      /// derived
      void BasisAtCenter(double* t1, double* t2, double* n) override
      {
        CORE::LINALG::Matrix<probdim, 1> t1_mat(t1, true);  // create view
        CORE::LINALG::Matrix<probdim, 1> t2_mat(t2, true);  // create view
        CORE::LINALG::Matrix<probdim, 1> n_mat(n, true);    // create view
        BasisAtCenter(t1_mat, t2_mat, n_mat);
      }
    };  // class ConcreteSide

    /*--------------------------------------------------------------------------*/
    class SideFactory
    {
     public:
      SideFactory(){};

      virtual ~SideFactory() = default;

      Side* create_side(CORE::FE::CellType sidetype, int sid, const std::vector<Node*>& nodes,
          const std::vector<Edge*>& edges) const;

     private:
      template <CORE::FE::CellType sidetype>
      CORE::GEO::CUT::Side* create_concrete_side(int sid, const std::vector<Node*>& nodes,
          const std::vector<Edge*>& edges, int probdim) const
      {
        Side* s = nullptr;
        // sanity check
        if (probdim < CORE::FE::dim<sidetype>)
          FOUR_C_THROW("Problem dimension is smaller than the side dimension!");

        switch (probdim)
        {
          case 2:
            s = new ConcreteSide<2, sidetype>(sid, nodes, edges);
            break;
          case 3:
            s = new ConcreteSide<3, sidetype>(sid, nodes, edges);
            break;
          default:
            FOUR_C_THROW("Unsupported problem dimension! (probdim = %d)", probdim);
            break;
        }
        return s;
      }
    };  // class SideFactory

  }  // namespace CUT
}  // namespace CORE::GEO

std::ostream& operator<<(std::ostream& stream, CORE::GEO::CUT::Side& s);

FOUR_C_NAMESPACE_CLOSE

#endif