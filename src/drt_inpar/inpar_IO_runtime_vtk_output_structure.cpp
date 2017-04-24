/*----------------------------------------------------------------------*/
/*!
\file inpar_IO_runtime_vtk_output_structure.cpp

\brief input parameters for VTK output of structural problem at runtime

\level 2

\maintainer Maximilian Grill
*/
/*----------------------------------------------------------------------*/

#include "inpar_IO_runtime_vtk_output_structure.H"

#include "drt_validparameters.H"
#include "inpar.H"
#include "inpar_parameterlist_utils.H"

#include <Teuchos_ParameterList.hpp>

namespace INPAR
{
namespace IO_RUNTIME_VTK_STRUCTURE
{

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
  void SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
  {
    using namespace DRT::INPUT;
    using Teuchos::tuple;
    using Teuchos::setStringToIntegralParameter;

    Teuchos::Array<std::string> yesnotuple = tuple<std::string>("Yes","No","yes","no","YES","NO");
    Teuchos::Array<int> yesnovalue = tuple<int>(true,false,true,false,true,false);

    // related sublist
    Teuchos::ParameterList& sublist_IO = list->sublist("IO",false,"");
    Teuchos::ParameterList& sublist_IO_VTK_structure =
        sublist_IO.sublist("RUNTIME VTK OUTPUT STRUCTURE",false,"");


    // output interval regarding steps: write output every INTERVAL_STEPS steps
    IntParameter( "INTERVAL_STEPS", -1,
        "write VTK output at runtime every INTERVAL_STEPS steps", &sublist_IO_VTK_structure );


    // data format for written numeric data
    setStringToIntegralParameter<int>(
      "OUTPUT_DATA_FORMAT", "binary", "data format for written numeric data",
      tuple<std::string>(
        "binary",
        "Binary",
        "ascii",
        "ASCII"),
      tuple<int>(
        INPAR::IO_RUNTIME_VTK_STRUCTURE::binary,
        INPAR::IO_RUNTIME_VTK_STRUCTURE::binary,
        INPAR::IO_RUNTIME_VTK_STRUCTURE::ascii,
        INPAR::IO_RUNTIME_VTK_STRUCTURE::ascii),
      &sublist_IO_VTK_structure);


    // whether to write output in every iteration of the nonlinear solver
    setStringToIntegralParameter<int>("EVERY_ITERATION","No",
                                 "write output in every iteration of the nonlinear solver",
                                 yesnotuple, yesnovalue, &sublist_IO_VTK_structure);

    // whether to write displacement state
    setStringToIntegralParameter<int>("DISPLACEMENT","No",
                                 "write displacement output",
                                 yesnotuple, yesnovalue, &sublist_IO_VTK_structure);

    // whether to write special output for beam elements
    setStringToIntegralParameter<int>("SPECIAL_OUTPUT_BEAMS","No",
                                 "write special output for beam elements",
                                 yesnotuple, yesnovalue, &sublist_IO_VTK_structure);
  }


}
}
