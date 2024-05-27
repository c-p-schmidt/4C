/*---------------------------------------------------------------------*/
/*! \file

\brief General utility methods for all Epetra objects


\level 1

*/
/*---------------------------------------------------------------------*/

#include "4C_utils_epetra_exceptions.hpp"

#include <Epetra_Object.h>

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void set_trace_back_mode(int tracebackmode)
{
  static const int global_mode = Epetra_Object::GetTracebackMode();
  Epetra_Object::SetTracebackMode(std::max(global_mode, tracebackmode));
}