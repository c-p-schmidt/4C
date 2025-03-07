// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_solver_nonlin.hpp"

#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Inpar::NlnSol::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;

  /*----------------------------------------------------------------------*
   * parameters for NOX - non-linear solution
   *----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs snox{"STRUCT NOX"};

  {
    std::vector<std::string> nonlinear_solver_valid_input = {"Line Search Based",
        "Pseudo Transient", "Trust Region Based", "Inexact Trust Region Based", "Tensor Based",
        "Single Step"};

    Core::Utils::string_parameter("Nonlinear Solver", "Line Search Based",
        "Choose a nonlinear solver method.", snox, nonlinear_solver_valid_input);
  }
  snox.move_into_collection(list);

  // sub-list direction
  Core::Utils::SectionSpecs direction{snox, "Direction"};

  {
    std::vector<std::string> newton_method_valid_input = {
        "Newton", "Steepest Descent", "NonlinearCG", "Broyden", "User Defined"};
    Core::Utils::string_parameter("Method", "Newton",
        "Choose a direction method for the nonlinear solver.", direction,
        newton_method_valid_input);

    std::vector<std::string> user_defined_method_valid_input = {"Newton", "Modified Newton"};
    Core::Utils::string_parameter("User Defined Method", "Modified Newton",
        "Choose a user-defined direction method.", direction, user_defined_method_valid_input);
  }
  direction.move_into_collection(list);

  // sub-sub-list "Newton"
  Core::Utils::SectionSpecs newton{direction, "Newton"};

  {
    std::vector<std::string> forcing_term_valid_input = {"Constant", "Type 1", "Type 2"};
    Core::Utils::string_parameter(
        "Forcing Term Method", "Constant", "", newton, forcing_term_valid_input);

    Core::Utils::double_parameter(
        "Forcing Term Initial Tolerance", 0.1, "initial linear solver tolerance", newton);
    Core::Utils::double_parameter("Forcing Term Minimum Tolerance", 1.0e-6, "", newton);
    Core::Utils::double_parameter("Forcing Term Maximum Tolerance", 0.01, "", newton);
    Core::Utils::double_parameter("Forcing Term Alpha", 1.5, "used only by \"Type 2\"", newton);
    Core::Utils::double_parameter("Forcing Term Gamma", 0.9, "used only by \"Type 2\"", newton);
    Core::Utils::bool_parameter("Rescue Bad Newton Solve", true,
        "If set to true, we will use "
        "the computed direction even if the linear solve does not achieve the tolerance "
        "specified by the forcing term",
        newton);
  }
  newton.move_into_collection(list);

  // sub-sub-list "Steepest Descent"
  Core::Utils::SectionSpecs steepestdescent{direction, "Steepest Descent"};

  {
    std::vector<std::string> scaling_type_valid_input = {
        "2-Norm", "Quadratic Model Min", "F 2-Norm", "None"};
    Core::Utils::string_parameter(
        "Scaling Type", "None", "", steepestdescent, scaling_type_valid_input);
  }
  steepestdescent.move_into_collection(list);

  // sub-list "Pseudo Transient"
  Core::Utils::SectionSpecs ptc{snox, "Pseudo Transient"};

  {
    Core::Utils::double_parameter("deltaInit", -1.0,
        "Initial time step size. If its negative, the initial time step is calculated "
        "automatically.",
        ptc);
    Core::Utils::double_parameter("deltaMax", std::numeric_limits<double>::max(),
        "Maximum time step size. "
        "If the new step size is greater than this value, the transient terms will be eliminated "
        "from the Newton iteration resulting in a full Newton solve.",
        ptc);
    Core::Utils::double_parameter("deltaMin", 1.0e-5, "Minimum step size.", ptc);
    Core::Utils::int_parameter(
        "Max Number of PTC Iterations", std::numeric_limits<int>::max(), "", ptc);
    Core::Utils::double_parameter("SER_alpha", 1.0, "Exponent of SET.", ptc);
    Core::Utils::double_parameter("ScalingFactor", 1.0, "Scaling Factor for ptc matrix.", ptc);

    std::vector<std::string> time_step_control_valid_input = {"SER",
        "Switched Evolution Relaxation", "TTE", "Temporal Truncation Error", "MRR",
        "Model Reduction Ratio"};
    Core::Utils::string_parameter(
        "Time Step Control", "SER", "", ptc, time_step_control_valid_input);

    std::vector<std::string> tsc_norm_type_valid_input = {"Two Norm", "One Norm", "Max Norm"};
    Core::Utils::string_parameter("Norm Type for TSC", "Max Norm",
        "Norm Type for the time step control", ptc, tsc_norm_type_valid_input);

    std::vector<std::string> scaling_op_valid_input = {
        "Identity", "CFL Diagonal", "Lumped Mass", "Element based"};
    Core::Utils::string_parameter("Scaling Type", "Identity",
        "Type of the scaling matrix for the PTC method.", ptc, scaling_op_valid_input);

    std::vector<std::string> build_scale_op_valid_input = {"every iter", "every timestep"};
    Core::Utils::string_parameter("Build scaling operator", "every timestep",
        "Build scaling operator in every iteration or timestep", ptc, build_scale_op_valid_input);
  }
  ptc.move_into_collection(list);

  // sub-list "Line Search"
  Core::Utils::SectionSpecs linesearch{snox, "Line Search"};

  {
    std::vector<std::string> method_valid_input = {
        "Full Step", "Backtrack", "Polynomial", "More'-Thuente", "User Defined"};
    Core::Utils::string_parameter("Method", "Full Step", "", linesearch, method_valid_input);


    Teuchos::Array<std::string> checktypes =
        Teuchos::tuple<std::string>("Complete", "Minimal", "None");
    Core::Utils::string_to_integral_parameter<::NOX::StatusTest::CheckType>(
        "Inner Status Test Check Type", "Minimal",
        "Specify the check type for the inner status tests.", checktypes,
        Teuchos::tuple<::NOX::StatusTest::CheckType>(
            ::NOX::StatusTest::Complete, ::NOX::StatusTest::Minimal, ::NOX::StatusTest::None),
        linesearch);
  }
  linesearch.move_into_collection(list);

  // sub-sub-list "Full Step"
  Core::Utils::SectionSpecs fullstep{linesearch, "Full Step"};

  {
    Core::Utils::double_parameter("Full Step", 1.0, "length of a full step", fullstep);
  }
  fullstep.move_into_collection(list);

  // sub-sub-list "Backtrack"
  Core::Utils::SectionSpecs backtrack{linesearch, "Backtrack"};

  {
    Core::Utils::double_parameter("Default Step", 1.0, "starting step length", backtrack);
    Core::Utils::double_parameter(
        "Minimum Step", 1.0e-12, "minimum acceptable step length", backtrack);
    Core::Utils::double_parameter("Recovery Step", 1.0,
        "step to take when the line search fails (defaults to value for \"Default Step\")",
        backtrack);
    Core::Utils::int_parameter(
        "Max Iters", 50, "maximum number of iterations (i.e., RHS computations)", backtrack);
    Core::Utils::double_parameter("Reduction Factor", 0.5,
        "A multiplier between zero and one that reduces the step size between line search "
        "iterations",
        backtrack);
    Core::Utils::bool_parameter("Allow Exceptions", false,
        "Set to true, if exceptions during the force evaluation and backtracking routine should be "
        "allowed.",
        backtrack);
  }
  backtrack.move_into_collection(list);

  // sub-sub-list "Polynomial"
  Core::Utils::SectionSpecs polynomial{linesearch, "Polynomial"};

  {
    Core::Utils::double_parameter("Default Step", 1.0, "Starting step length", polynomial);
    Core::Utils::int_parameter("Max Iters", 100,
        "Maximum number of line search iterations. "
        "The search fails if the number of iterations exceeds this value",
        polynomial);
    Core::Utils::double_parameter("Minimum Step", 1.0e-12,
        "Minimum acceptable step length. The search fails if the computed $\\lambda_k$ "
        "is less than this value",
        polynomial);

    std::vector<std::string> recovery_step_type_valid_input = {"Constant", "Last Computed Step"};
    Core::Utils::string_parameter("Recovery Step Type", "Constant",
        "Determines the step size to take when the line search fails", polynomial,
        recovery_step_type_valid_input);

    Core::Utils::double_parameter("Recovery Step", 1.0,
        "The value of the step to take when the line search fails. Only used if the \"Recovery "
        "Step Type\" is set to \"Constant\"",
        polynomial);

    std::vector<std::string> interpolation_type_valid_input = {"Quadratic", "Quadratic3", "Cubic"};
    Core::Utils::string_parameter("Interpolation Type", "Cubic",
        "Type of interpolation that should be used", polynomial, interpolation_type_valid_input);

    Core::Utils::double_parameter("Min Bounds Factor", 0.1,
        "Choice for $\\gamma_{\\min}$, i.e., the factor that limits the minimum size "
        "of the new step based on the previous step",
        polynomial);
    Core::Utils::double_parameter("Max Bounds Factor", 0.5,
        "Choice for $\\gamma_{\\max}$, i.e., the factor that limits the maximum size "
        "of the new step based on the previous step",
        polynomial);

    std::vector<std::string> sufficient_decrease_condition_valid_input = {
        "Armijo-Goldstein", "Ared/Pred", "None"};
    Core::Utils::string_parameter("Sufficient Decrease Condition", "Armijo-Goldstein",
        "Choice to use for the sufficient decrease condition", polynomial,
        sufficient_decrease_condition_valid_input);

    Core::Utils::double_parameter(
        "Alpha Factor", 1.0e-4, "Parameter choice for sufficient decrease condition", polynomial);
    Core::Utils::bool_parameter("Force Interpolation", false,
        "Set to true if at least one interpolation step should be used. The default is false which "
        "means that the line search will stop if the default step length satisfies the convergence "
        "criteria",
        polynomial);
    Core::Utils::bool_parameter("Use Counters", true,
        "Set to true if we should use counters and then output the result to the parameter list as "
        "described in Output Parameters",
        polynomial);
    Core::Utils::int_parameter("Maximum Iteration for Increase", 0,
        "Maximum index of the nonlinear iteration for which we allow a relative increase",
        polynomial);
    Core::Utils::double_parameter("Allowed Relative Increase", 100, "", polynomial);
  }
  polynomial.move_into_collection(list);

  // sub-sub-list "More'-Thuente"
  Core::Utils::SectionSpecs morethuente{linesearch, "More'-Thuente"};

  {
    Core::Utils::double_parameter("Sufficient Decrease", 1.0e-4,
        "The ftol in the sufficient decrease condition", morethuente);
    Core::Utils::double_parameter(
        "Curvature Condition", 0.9999, "The gtol in the curvature condition", morethuente);
    Core::Utils::double_parameter("Interval Width", 1.0e-15,
        "The maximum width of the interval containing the minimum of the modified function",
        morethuente);
    Core::Utils::double_parameter(
        "Maximum Step", 1.0e6, "maximum allowable step length", morethuente);
    Core::Utils::double_parameter(
        "Minimum Step", 1.0e-12, "minimum allowable step length", morethuente);
    Core::Utils::int_parameter("Max Iters", 20,
        "maximum number of right-hand-side and corresponding Jacobian evaluations", morethuente);
    Core::Utils::double_parameter("Default Step", 1.0, "starting step length", morethuente);

    std::vector<std::string> recovery_step_type_valid_input = {"Constant", "Last Computed Step"};
    Core::Utils::string_parameter("Recovery Step Type", "Constant",
        "Determines the step size to take when the line search fails", morethuente,
        recovery_step_type_valid_input);

    Core::Utils::double_parameter("Recovery Step", 1.0,
        "The value of the step to take when the line search fails. Only used if the \"Recovery "
        "Step Type\" is set to \"Constant\"",
        morethuente);

    std::vector<std::string> sufficient_decrease_condition_valid_input = {
        "Armijo-Goldstein", "Ared/Pred", "None"};
    Core::Utils::string_parameter("Sufficient Decrease Condition", "Armijo-Goldstein",
        "Choice to use for the sufficient decrease condition", morethuente,
        sufficient_decrease_condition_valid_input);

    Core::Utils::bool_parameter("Optimize Slope Calculation", false,
        "Boolean value. If set to true the value of $s^T J^T F$ is estimated using a "
        "directional derivative in a call to ::NOX::LineSearch::Common::computeSlopeWithOutJac. "
        "If false the slope computation is computed with the "
        "::NOX::LineSearch::Common::computeSlope method. "
        "Setting this to true eliminates having to compute the Jacobian at each inner iteration of "
        "the More'-Thuente line search",
        morethuente);
  }
  morethuente.move_into_collection(list);

  // sub-list "Trust Region"
  Core::Utils::SectionSpecs trustregion{snox, "Trust Region"};

  {
    Core::Utils::double_parameter("Minimum Trust Region Radius", 1.0e-6,
        "Minimum allowable trust region radius", trustregion);
    Core::Utils::double_parameter("Maximum Trust Region Radius", 1.0e+9,
        "Maximum allowable trust region radius", trustregion);
    Core::Utils::double_parameter("Minimum Improvement Ratio", 1.0e-4,
        "Minimum improvement ratio to accept the step", trustregion);
    Core::Utils::double_parameter("Contraction Trigger Ratio", 0.1,
        "If the improvement ratio is less than this value, then the trust region is contracted by "
        "the amount specified by the \"Contraction Factor\". Must be larger than \"Minimum "
        "Improvement Ratio\"",
        trustregion);
    Core::Utils::double_parameter("Contraction Factor", 0.25, "", trustregion);
    Core::Utils::double_parameter("Expansion Trigger Ratio", 0.75,
        "If the improvement ratio is greater than this value, then the trust region is contracted "
        "by the amount specified by the \"Expansion Factor\"",
        trustregion);
    Core::Utils::double_parameter("Expansion Factor", 4.0, "", trustregion);
    Core::Utils::double_parameter("Recovery Step", 1.0, "", trustregion);
  }
  trustregion.move_into_collection(list);

  // sub-list "Printing"
  Core::Utils::SectionSpecs printing{snox, "Printing"};

  {
    Core::Utils::bool_parameter("Error", false, "", printing);
    Core::Utils::bool_parameter("Warning", true, "", printing);
    Core::Utils::bool_parameter("Outer Iteration", true, "", printing);
    Core::Utils::bool_parameter("Inner Iteration", true, "", printing);
    Core::Utils::bool_parameter("Parameters", false, "", printing);
    Core::Utils::bool_parameter("Details", false, "", printing);
    Core::Utils::bool_parameter("Outer Iteration StatusTest", true, "", printing);
    Core::Utils::bool_parameter("Linear Solver Details", false, "", printing);
    Core::Utils::bool_parameter("Test Details", false, "", printing);
    Core::Utils::bool_parameter("Debug", false, "", printing);
  }
  printing.move_into_collection(list);

  // sub-list "Status Test"
  Core::Utils::SectionSpecs statusTest{snox, "Status Test"};

  {
    statusTest.specs.emplace_back(
        Core::IO::InputSpecBuilders::parameter<Core::IO::Noneable<std::filesystem::path>>(
            "XML File",
            {.description = "Filename of XML file with configuration of nox status test",
                .default_value = Core::IO::Noneable<std::filesystem::path>()}));
  }
  statusTest.move_into_collection(list);

  // sub-list "Solver Options"
  Core::Utils::SectionSpecs solverOptions{snox, "Solver Options"};

  {
    Teuchos::Array<std::string> meritFct = Teuchos::tuple<std::string>("Sum of Squares");
    Core::Utils::string_to_integral_parameter<NOX::Nln::MeritFunction::MeritFctName>(
        "Merit Function", "Sum of Squares", "", meritFct,
        Teuchos::tuple<NOX::Nln::MeritFunction::MeritFctName>(
            NOX::Nln::MeritFunction::mrtfct_sum_of_squares),
        solverOptions);

    std::vector<std::string> status_test_check_type_valid_input = {"Complete", "Minimal", "None"};
    Core::Utils::string_parameter("Status Test Check Type", "Complete", "", solverOptions,
        status_test_check_type_valid_input);
  }
  solverOptions.move_into_collection(list);

  // sub-sub-sub-list "Linear Solver"
  Core::Utils::SectionSpecs linearSolver{newton, "Linear Solver"};

  {
    // convergence criteria adaptivity
    Core::Utils::bool_parameter("Adaptive Control", false,
        "Switch on adaptive control of linear solver tolerance for nonlinear solution",
        linearSolver);
    Core::Utils::double_parameter("Adaptive Control Objective", 0.1,
        "The linear solver shall be this much better than the current nonlinear residual in the "
        "nonlinear convergence limit",
        linearSolver);
    Core::Utils::bool_parameter(
        "Zero Initial Guess", true, "Zero out the delta X vector if requested.", linearSolver);
    Core::Utils::bool_parameter("Computing Scaling Manually", false,
        "Allows the manually scaling of your linear system (not supported at the moment).",
        linearSolver);
    Core::Utils::bool_parameter(
        "Output Solver Details", true, "Switch the linear solver output on and off.", linearSolver);
  }
  linearSolver.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
