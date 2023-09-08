#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
    #if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace dealii::LinearAlgebraPETSc;
    #  define USE_PETSC_LA
    #elif defined(DEAL_II_WITH_TRILINOS)
    using namespace dealii::LinearAlgebraTrilinos;
    #else
    #  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
    #endif
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/grid/manifold.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/sparse_ilu.h>

#include <iostream>
#include <fstream>
#include <sstream>
namespace Navierstokes
{
    using namespace dealii;

    template <int dim>
    class BoundaryValues104;
    
    template <int dim>
    class StokesProblem
    {        
    private:
        MPI_Comm                                  mpi_communicator;
        double deltat = 0.005;
        double totaltime = 15;
        double viscosity = 0.89, density = 500.0;

        int meshrefinement = 2;
        int degree;
        parallel::distributed::Triangulation<dim> triangulation;
        LA::MPI::SparseMatrix                     ns_system_matrix;
        DoFHandler<dim>                           dof_handler;
        FESystem<dim>                             fe;
        LA::MPI::Vector                           lr_solution, lr_old_solution;
        LA::MPI::Vector                           lo_system_rhs; 
        AffineConstraints<double>                 stokesconstraints;
        IndexSet                                  owned_partitioning_stokes;
        IndexSet                                  relevant_partitioning_stokes;
        ConditionalOStream                        pcout;
        TimerOutput                               computing_timer;
        
    public:
        void setup_stokessystem();
        void resetup_stokessystem(const BoundaryValues104<dim>);
        void assemble_stokessystem();
        void apply_boundary_conditions_and_rhs();
        void solve_stokes();
        void output_results (int );
        void timeloop();
        
        StokesProblem(int degreein)
        :
        mpi_communicator (MPI_COMM_WORLD),
        degree(degreein),
        triangulation (mpi_communicator),
        dof_handler(triangulation),
        fe(FE_Q<dim>(degree+1), dim, FE_Q<dim>(degree), 1),
        pcout (std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        computing_timer (mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
        {      
            pcout << "stokes constructor success...."<< std::endl;
        }
    };
    //=====================================================  
    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
        RightHandSide () : Function<dim>(dim+1)
        {}
        virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;
    };
        
    template <int dim>
    void
    RightHandSide<dim>::vector_value(const Point<dim> &p,  Vector<double> &values) const
    {
        values[0] = 0;
        values[1] = 0;
        values[2] = 0*p[0];
    }
    //===============================================
    template<int dim>
    class BoundaryValues104 : public Function<dim>
    {
    public:
        BoundaryValues104():Function<dim>(dim+1)
        {}
        
        virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;
    };
    
    template<int dim>
    void BoundaryValues104<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
    {
        const double time = this->get_time();
        
        if(time < 2.5)
        {
            //             values[0] = time*p[1]*(2-p[1])*p[0]*(2-p[0]);
            values[0] = time*p[1]*(2-p[1]);
            values[1] = 0*p.square();
            values[2] = 0;
        }
        else
        {
            //             values[0] = 2.5*p[1]*(2-p[1])*p[0]*(2-p[0]);
            values[0] = 2.5*p[1]*(2-p[1]);
            values[1] = 0*p.square();
            values[2] = 0;
        }
    }
    //==============================================================  
    template <int dim>
    void StokesProblem<dim>::setup_stokessystem()
    {  
        TimerOutput::Scope t(computing_timer, "setup_stokessystem");
        pcout <<"in setup_stokessystem "<<std::endl;
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file("channel_2D_hex.msh");
        grid_in.read_msh(input_file);
        triangulation.refine_global (meshrefinement);
        dof_handler.distribute_dofs(fe);
        
        pcout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Total number of cells: "
        << triangulation.n_cells()
        << std::endl;
        pcout << "   Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl;
        
        std::vector<unsigned int> block_component(dim+1,0);
        block_component[dim] = 1;
//         std::vector<types::global_dof_index> dofs_per_block (2);        
        const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
//         DoFTools::count_dofs_per_fe_block(dof_handler, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0],  n_p = dofs_per_block[1];
        pcout << " (" << n_u << '+' << n_p << ')' << std::endl;
        pcout << "dofspercell "<< fe.dofs_per_cell << std::endl;
        
        owned_partitioning_stokes = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (dof_handler, relevant_partitioning_stokes);
        
        BoundaryValues104<dim> boundaryvalues104;
        
        {
            stokesconstraints.reinit (relevant_partitioning_stokes);
            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim);
            ComponentMask velocities_mask = fe.component_mask(velocities);
            ComponentMask pressure_mask = fe.component_mask(pressure);
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (101);
            no_normal_flux_boundaries.insert (103);
            no_normal_flux_boundaries.insert (105);
            no_normal_flux_boundaries.insert (106);
            no_normal_flux_boundaries.insert (107);
            no_normal_flux_boundaries.insert (108);
//             VectorTools::compute_no_normal_flux_constraints (dof_handler, 0, no_normal_flux_boundaries, stokesconstraints);
            VectorTools::interpolate_boundary_values (dof_handler, 101, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 103, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 104, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 105, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 106, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 107, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 108, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 102, ZeroFunction<dim>(dim+1), stokesconstraints, pressure_mask);
            std::set<types::boundary_id> no_tangential_flux_boundaries;
            no_tangential_flux_boundaries.insert (102);
            VectorTools::compute_normal_flux_constraints(dof_handler, 0, no_tangential_flux_boundaries, stokesconstraints); 
            stokesconstraints.close();
        }
        
        ns_system_matrix.clear();        
        DynamicSparsityPattern dsp (relevant_partitioning_stokes);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, stokesconstraints, false);
        SparsityTools::distribute_sparsity_pattern (dsp, dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_stokes);
        ns_system_matrix.reinit (owned_partitioning_stokes, owned_partitioning_stokes, dsp, mpi_communicator);
        lr_solution.reinit(owned_partitioning_stokes, relevant_partitioning_stokes, mpi_communicator);
        lr_old_solution.reinit(owned_partitioning_stokes, relevant_partitioning_stokes, mpi_communicator);
        lo_system_rhs.reinit(owned_partitioning_stokes, mpi_communicator);
        pcout <<"end of setup_stokessystem "<<std::endl;
    }
    //================================================
    template <int dim>
    void StokesProblem<dim>::resetup_stokessystem(const BoundaryValues104<dim> boundaryvalues104)
    {        
        {
            stokesconstraints.reinit (relevant_partitioning_stokes);
            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim);
            ComponentMask velocities_mask = fe.component_mask(velocities);
            ComponentMask pressure_mask = fe.component_mask(pressure);
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (101);
            no_normal_flux_boundaries.insert (103);
            no_normal_flux_boundaries.insert (105);
            no_normal_flux_boundaries.insert (106);
            no_normal_flux_boundaries.insert (107);
            no_normal_flux_boundaries.insert (108);
//             VectorTools::compute_no_normal_flux_constraints (dof_handler, 0, no_normal_flux_boundaries, stokesconstraints);
            VectorTools::interpolate_boundary_values (dof_handler, 104, boundaryvalues104, stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 101, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 103, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 105, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 106, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 107, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 108, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            VectorTools::interpolate_boundary_values (dof_handler, 102, ZeroFunction<dim>(dim+1), stokesconstraints, pressure_mask);
            std::set<types::boundary_id> no_tangential_flux_boundaries;
            no_tangential_flux_boundaries.insert (102);
            VectorTools::compute_normal_flux_constraints(dof_handler, 0, no_tangential_flux_boundaries, stokesconstraints); 
            stokesconstraints.close();
        }
    }
    
    //============================================================
 template <int dim>
    void StokesProblem<dim>::assemble_stokessystem()
    {
        TimerOutput::Scope t(computing_timer, "assemble_stokessystem");
        pcout <<"assemble_stokessystem "<<std::endl;
        ns_system_matrix=0;
        lo_system_rhs=0;
        
        QGauss<dim>   quadrature_formula(degree+2);
        QGauss<dim-1> face_quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);        
        
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);
        
        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        //         const unsigned int   vof_dofs_per_cell   = fevof.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim); 
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index>  local_dof_indices (dofs_per_cell);        
        const RightHandSide<dim>              right_hand_side;
        std::vector<Vector<double>>           rhs_values(n_q_points, Vector<double>(dim+1));
        std::vector<Vector<double>>           neumann_boundary_values(n_face_q_points, Vector<double>(dim+1));        
        std::vector<Tensor<1, dim>>           value_phi_u (dofs_per_cell);    
        std::vector<Tensor<2, dim>>           gradient_phi_u (dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>>  symgrad_phi_u (dofs_per_cell);
        std::vector<double>                   div_phi_u(dofs_per_cell);
        std::vector<double>                   phi_p(dofs_per_cell);        
        std::vector<Tensor<2, dim> >          old_solution_gradients(n_q_points);
        std::vector<Tensor<1, dim> >          old_solution_values(n_q_points);
        
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
        
        for (; cell!=endc; ++cell)
        { 
            if (cell->is_locally_owned())
            {
                fe_values.reinit (cell);
                fe_values[velocities].get_function_values(lr_old_solution, old_solution_values);
                fe_values[velocities].get_function_gradients(lr_old_solution, old_solution_gradients);
                
                local_matrix = 0;
                local_rhs = 0;
                right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
                
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {                        
                        value_phi_u[k]   = fe_values[velocities].value (k, q);
                        gradient_phi_u[k]= fe_values[velocities].gradient (k, q);
                        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                        div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                        phi_p[k]         = fe_values[pressure].value (k, q);
                    }
                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {                    
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {
                            local_matrix(i, j) += ((value_phi_u[i]*value_phi_u[j] + 
                            deltat * value_phi_u[i] * (gradient_phi_u[j] * (old_solution_values[q])) +
                            (2*deltat*viscosity/density)*scalar_product(symgrad_phi_u[i], symgrad_phi_u[j])) -
                            deltat * div_phi_u[i] * phi_p[j]/density - 
                            phi_p[i] * div_phi_u[j]) *
                            fe_values.JxW(q);
                        }
                        const unsigned int component_i = fe.system_to_component_index(i).first;
//                         local_rhs(i) += deltat*(fe_values.shape_value(i,q) * rhs_values[q](component_i) + value_phi_u[i] * (old_solution_gradients[q] * (old_solution_values[q]))) * fe_values.JxW(q);
local_rhs(i) += (deltat*fe_values.shape_value(i,q)*rhs_values[q](component_i) + old_solution_values[q]*value_phi_u[i]) * fe_values.JxW(q);
                    } // end of i loop                
                }  // end of quadrature points loop                
                cell->get_dof_indices (local_dof_indices);         
                stokesconstraints.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, ns_system_matrix, lo_system_rhs);
            } // end of if cell->locally owned
        } // end of cell loop
        ns_system_matrix.compress (VectorOperation::add);
        lo_system_rhs.compress (VectorOperation::add);
//         pcout <<"end of assemble_stokessystem "<<std::endl;
    } // end of assemble system
    //================================================================
    template <int dim>
    void StokesProblem<dim>::solve_stokes()
    {
        pcout <<"solve_stokes"<<std::endl;
        TimerOutput::Scope t(computing_timer, "solve_stokes");
        LA::MPI::Vector  distributed_solution_stokes (owned_partitioning_stokes, mpi_communicator);
        
        SolverControl solver_control_stokes (dof_handler.n_dofs(), 1e-12);
        dealii::PETScWrappers::SparseDirectMUMPS solver_stokes(solver_control_stokes, mpi_communicator);
        
        solver_stokes.solve (ns_system_matrix, distributed_solution_stokes, lo_system_rhs);
        stokesconstraints.distribute(distributed_solution_stokes);                 
        lr_solution = distributed_solution_stokes;
        
        lr_old_solution = lr_solution;
//         pcout <<"end of solve_stokes "<<std::endl;
    }
    //================================================================
    template <int dim>
    void StokesProblem<dim>::output_results(int timestepnumber)
    {
        TimerOutput::Scope t(computing_timer, "output");
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.emplace_back ("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (lr_solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
        
        Vector<float> subdomain (triangulation.n_active_cells());
        for (unsigned int i=0; i<subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector (subdomain, "subdomain");
//         data_out.add_data_vector(vof_dof_handler, lr_old_vof_solution, "vof");
        data_out.build_patches ();
        
        std::string filenamebase = "znoslip-";
        
        const std::string filename = (filenamebase + Utilities::int_to_string (timestepnumber, 3) + "." +Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
        std::ofstream output ((filename + ".vtu").c_str());
        data_out.write_vtu (output);
        
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            std::vector<std::string> filenames;
            for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
                filenames.push_back (filenamebase + Utilities::int_to_string (timestepnumber, 3) + "." + Utilities::int_to_string (i, 4) + ".vtu");
            
            std::ofstream master_output ((filenamebase + Utilities::int_to_string (timestepnumber, 3) + ".pvtu").c_str());
            data_out.write_pvtu_record (master_output, filenames);
        }
    }

    //==================================================================  
    template <int dim>
    void StokesProblem<dim>::timeloop()
    {      
        double timet = deltat;
        int timestepnumber=0;
        BoundaryValues104<dim> boundaryvalues104;
        
        while(timet<totaltime)
        {  
            output_results(timestepnumber);
            boundaryvalues104.set_time(timet);
            resetup_stokessystem(boundaryvalues104);
            assemble_stokessystem();
            solve_stokes();
            pcout <<"timet "<<timet <<std::endl;                       
            timet+=deltat;
            timestepnumber++;
        } 
        output_results(timestepnumber);
    }
}  // end of namespace
//====================================================
int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace Navierstokes;
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);        
        StokesProblem<2> flow_problem(1);   
        flow_problem.setup_stokessystem();
        flow_problem.timeloop();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }    
    return 0;
}
