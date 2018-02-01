require 'mesa_script'

Inlist.make_inlist('inlist_1Msun_tau1') do
  # inlist to evolve a 15 solar mass star
  
  # For the sake of future readers of this file (yourself included),
  # ONLY include the controls you are actually using.  DO NOT include
  # all of the other controls that simply have their default values.
  
  # &star_job
  
    # begin with a pre-main sequence model
      create_pre_main_sequence_model true
  
    # save a model at the end of the run
      save_model_when_terminate false
      save_model_filename '15M_at_TAMS.mod'
  
    # display on-screen plots
      pgstar_flag true
  
  # / !end of star_job namelist
  
  
  # &controls
  
    # starting specifications
      initial_mass 15 # in Msun units
  
    # stop when the star nears ZAMS (Lnuc/L > 0.99)
      lnuc_div_l_zams_limit 0.99e0
      stop_near_zams true
  
    # stop when the center mass fraction of h1 drops below this limit
      xa_central_lower_limit_species[1] = 'h1'
      xa_central_lower_limit[1] = 1e-3
  
  # / ! end of controls namelist
end
