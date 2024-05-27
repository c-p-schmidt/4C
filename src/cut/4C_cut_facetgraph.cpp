/*---------------------------------------------------------------------*/
/*! \file

\brief graph to create volume cells from facets and lines

\level 3


*----------------------------------------------------------------------*/

#include "4C_cut_facetgraph.hpp"

#include "4C_cut_facetgraph_simple.hpp"
#include "4C_cut_mesh.hpp"
#include "4C_cut_output.hpp"
#include "4C_cut_side.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
CORE::GEO::CUT::FacetGraph::FacetGraph(
    const std::vector<Side *> &sides, const plain_facet_set &facets)
    : graph_(facets.size())
{
  std::map<std::pair<Point *, Point *>, plain_facet_set> lines;
  for (plain_facet_set::const_iterator i = facets.begin(); i != facets.end(); ++i)
  {
    Facet *f = *i;
    f->GetLines(lines);

    // connect side facet to all hole lines

    if (f->HasHoles())
    {
      const plain_facet_set &holes = f->Holes();

      // NOTE: Adding hole facet to the hole_lines is already done in the previous step since
      // f->GetLines() also checks for the nested holes so this might be redundant
      std::map<std::pair<Point *, Point *>, plain_facet_set> hole_lines;
      for (plain_facet_set::const_iterator i = holes.begin(); i != holes.end(); ++i)
      {
        Facet *h = *i;
        h->GetLines(hole_lines);
      }
      for (std::map<std::pair<Point *, Point *>, plain_facet_set>::iterator i = hole_lines.begin();
           i != hole_lines.end(); ++i)
      {
        const std::pair<Point *, Point *> &l = i->first;
        lines[l].insert(f);
      }
    }
  }

  plain_facet_set all_facets = facets;
  for (plain_facet_set::const_iterator i = facets.begin(); i != facets.end(); ++i)
  {
    Facet *f = *i;
    if (f->HasHoles())
    {
      const plain_facet_set &holes = f->Holes();
      std::copy(holes.begin(), holes.end(), std::inserter(all_facets, all_facets.begin()));
    }
  }

  // fix for very rare case
  for (std::map<std::pair<Point *, Point *>, plain_facet_set>::iterator li = lines.begin();
       li != lines.end();)
  {
    plain_facet_set &fs = li->second;
    if (fs.size() < 2)
    {
      // for some unknown reason this does not lead to anything critical.., so error is left out
      // FOUR_C_THROW("Removing this error will most probably lead to error is facetgraph later on
      // anyway. Consider analyzing your case!");
      for (plain_facet_set::iterator i = fs.begin(); i != fs.end(); ++i)
      {
        Facet *f = *i;
        all_facets.erase(f);
      }
      lines.erase(li++);
    }
    else
    {
      ++li;
    }
  }

  graph_.SetSplit(all_facets.size());

  all_facets_.reserve(all_facets.size());
  all_facets_.assign(all_facets.begin(), all_facets.end());

  all_lines_.reserve(lines.size());

  // Mapping between indexes in colouredgraph and corresponding pointers of facets or lines
  std::map<int, const void *> index_value_map;

  for (std::map<std::pair<Point *, Point *>, plain_facet_set>::iterator i = lines.begin();
       i != lines.end(); ++i)
  {
    const std::pair<Point *, Point *> &l = i->first;
    const plain_facet_set &fs = i->second;
    // Because graph_ ids organized as follows -> [  facets |    lines ]
    //                                             ^split = all_facets.size()
    // so for lines we operate on indexes in the second part of the array

    int current = all_facets_.size() + all_lines_.size();
    index_value_map[current] = static_cast<const void *>(&(i->first));
    all_lines_.push_back(l);

    for (plain_facet_set::const_iterator i = fs.begin(); i != fs.end(); ++i)
    {
      Facet *f = *i;
      if (all_facets.count(f) > 0)
      {
        graph_.Add(FacetId(f), current);
        index_value_map[FacetId(f)] = static_cast<const void *>(f);
      }
    }
  }

  // graph is the graph of connected lines and facets,  including internal facets
  // cycle is the graph of connected line and facets, but only the outer ones, without internals

  graph_.TestClosed();

  COLOREDGRAPH::Graph cycle(all_facets_.size());

  for (std::vector<Side *>::const_iterator i = sides.begin(); i != sides.end(); ++i)
  {
    Side *s = *i;
    const std::vector<Facet *> &side_facets = s->Facets();

    for (std::vector<Facet *>::const_iterator i = side_facets.begin(); i != side_facets.end(); ++i)
    {
      Facet *f = *i;

      if (all_facets.count(f) > 0)
      {
        int p1 = FacetId(f);
        plain_int_set &row = graph_[p1];
        for (plain_int_set::iterator i = row.begin(); i != row.end(); ++i)
        {
          int p2 = *i;
          cycle.Add(p1, p2);
        }
      }

      if (f->HasHoles())
      {
        const plain_facet_set &hs = f->Holes();
        for (plain_facet_set::const_iterator i = hs.begin(); i != hs.end(); ++i)
        {
          Facet *h = *i;

          if (all_facets.count(h) > 0)
          {
            int p1 = FacetId(h);
            plain_int_set &row = graph_[p1];
            for (plain_int_set::const_iterator i = row.begin(); i != row.end(); ++i)
            {
              int p2 = *i;
              cycle.Add(p1, p2);
            }
          }
        }
      }
    }
  }

  // Free are cut lines and cut facets, that are inside ( e.g. cut_side cut facets inside the
  // element)
  plain_int_set free;
  graph_.GetAll(free);

  for (COLOREDGRAPH::Graph::const_iterator i = cycle.begin(); i != cycle.end(); ++i)
  {
    int p = i->first;
    free.erase(p);
  }

  // used is the external and it is looped over already , free is the internal, but we still use the
  // main graph_ with everything
  COLOREDGRAPH::Graph used(cycle);
  graph_.Map(&index_value_map);
  cycle_list_.AddPoints(graph_, used, cycle, free, all_lines_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CORE::GEO::CUT::FacetGraph::CreateVolumeCells(
    Mesh &mesh, Element *element, plain_volumecell_set &cells)
{
  std::vector<plain_facet_set> volumes;
  volumes.reserve(cycle_list_.size());

  std::vector<int> counter(all_facets_.size(), 0);

  for (COLOREDGRAPH::CycleList::iterator i = cycle_list_.begin(); i != cycle_list_.end(); ++i)
  {
    COLOREDGRAPH::Graph &g = *i;

    volumes.push_back(plain_facet_set());
    plain_facet_set &collected_facets = volumes.back();

    for (COLOREDGRAPH::Graph::const_iterator i = g.begin(); i != g.end(); ++i)
    {
      int p = i->first;
      if (p >= g.Split())
      {
        break;
      }
      collected_facets.insert(all_facets_[p]);
      counter[p] += 1;
    }
  }

  for (std::vector<int>::iterator i = counter.begin(); i != counter.end(); ++i)
  {
    int p = *i;
    if (p > 2)
    {
      FOUR_C_THROW("Check the case!");
      // assume this is a degenerated cell we do not need to build
      return;
    }
  }

  // finally add the volumes to the volume cells
  AddToVolumeCells(mesh, element, volumes, cells);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CORE::GEO::CUT::FacetGraph::AddToVolumeCells(Mesh &mesh, Element *element,
    std::vector<plain_facet_set> &volumes, plain_volumecell_set &cells) const
{
  for (std::vector<plain_facet_set>::iterator i = volumes.begin(); i != volumes.end(); ++i)
  {
    plain_facet_set &collected_facets = *i;

    // check facet number
    if (collected_facets.size() < (element->Dim() + 1))
    {
      Print();

      int fsc = 0;
      for (std::vector<plain_facet_set>::const_iterator fs = volumes.begin(); fs != volumes.end();
           ++fs)
        OUTPUT::GmshFacetsOnly(*fs, element, fsc++);

      std::ofstream file_element("add_to_volume_cells_facetgraph_failed.pos");
      CORE::GEO::CUT::OUTPUT::GmshElementDump(file_element, element, false);
      file_element.close();

      FOUR_C_THROW(
          "The facet number is too small to represent a volume cell! \n"
          "If this happens, it is an indication for missing internal facets.");
    }

    std::map<std::pair<Point *, Point *>, plain_facet_set> volume_lines;
    CollectVolumeLines(collected_facets, volume_lines);

    cells.insert(mesh.NewVolumeCell(collected_facets, volume_lines, element));
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CORE::GEO::CUT::FacetGraph::CollectVolumeLines(plain_facet_set &collected_facets,
    std::map<std::pair<Point *, Point *>, plain_facet_set> &volume_lines) const
{
  for (plain_facet_set::iterator i = collected_facets.begin(); i != collected_facets.end(); ++i)
  {
    Facet *f = *i;
    f->GetLines(volume_lines);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::GEO::CUT::FacetGraph> CORE::GEO::CUT::FacetGraph::Create(
    const std::vector<Side *> &sides, const plain_facet_set &facets)
{
  Teuchos::RCP<FacetGraph> fg = Teuchos::null;


  // get current underlying element dimension
  const unsigned dim = sides[0]->Elements()[0]->Dim();
  switch (dim)
  {
    case 1:
      fg = Teuchos::rcp(new SimpleFacetGraph1D(sides, facets));
      break;
    case 2:
      fg = Teuchos::rcp(new SimpleFacetGraph2D(sides, facets));
      break;
    case 3:
      fg = Teuchos::rcp(new FacetGraph(sides, facets));
      break;
    default:
      FOUR_C_THROW("Unsupported element dimension!");
      break;
  }

  // return the facet graph object pointer
  return fg;
};

FOUR_C_NAMESPACE_CLOSE