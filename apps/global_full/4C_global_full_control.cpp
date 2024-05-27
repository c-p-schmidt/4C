/*----------------------------------------------------------------------*/
/*! \file

\brief the basic 4C routine

\level 0


*/
/*----------------------------------------------------------------------*/

#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_global_full_init_control.hpp"
#include "4C_global_full_inp_control.hpp"
#include "4C_io_pstream.hpp"

void ntacal();

/*----------------------------------------------------------------------*
 | main routine                                           m.gee 8/00    |
 *----------------------------------------------------------------------*/
void ntam(int argc, char *argv[])
{
  using namespace FourC;

  Teuchos::RCP<Epetra_Comm> gcomm = GLOBAL::Problem::Instance()->GetCommunicators()->GlobalComm();

  double t0, ti, tc;

  /// IO file names and kenners
  std::string inputfile_name;
  std::string outputfile_kenner;
  std::string restartfile_kenner;

  ntaini_ccadiscret(argc, argv, inputfile_name, outputfile_kenner, restartfile_kenner);

  /* input phase, input of all information */

  t0 = GLOBAL::Problem::Walltime();

  ntainp_ccadiscret(inputfile_name, outputfile_kenner, restartfile_kenner);

  ti = GLOBAL::Problem::Walltime() - t0;
  if (gcomm->MyPID() == 0)
  {
    IO::cout << "\nTotal CPU Time for INPUT:       " << std::setw(10) << std::setprecision(3)
             << std::scientific << ti << " sec \n\n";
  }

  /*--------------------------------------------------calculation phase */
  t0 = GLOBAL::Problem::Walltime();

  ntacal();

  tc = GLOBAL::Problem::Walltime() - t0;
  if (gcomm->MyPID() == 0)
  {
    IO::cout << "\nTotal CPU Time for CALCULATION: " << std::setw(10) << std::setprecision(3)
             << std::scientific << tc << " sec \n\n";
  }
}