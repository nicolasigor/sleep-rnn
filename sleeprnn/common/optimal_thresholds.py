import os


OPTIMAL_THR_FOR_CKPT_DICT = {
    os.path.join('20190504_bsf_wn_train_mass_ss', 'bsf'): [0.64, 0.52, 0.52, 0.48],
    os.path.join('20190504_bsf_wn_train_mass_kc', 'bsf'): [0.52, 0.56, 0.54, 0.56],
    os.path.join('20190506_bsf_n2_train_mass_ss', 'bsf'): [0.52, 0.46, 0.50, 0.50],
    os.path.join('20190506_bsf_n2_train_mass_kc', 'bsf'): [0.52, 0.52, 0.56, 0.46],
    os.path.join('20190516_bsf_n2_train_inta_ss', 'bsf'): [0.48, 0.52, 0.48, 0.44],
    os.path.join('20190516_bsf_v2_n2_train_inta_ss', 'bsf'): [0.46, 0.52, 0.50, 0.46],
    os.path.join('20190522_bsf_newer_wins_fix_n2_train_inta_ss', 'bsf'): [0.44, 0.5, 0.44, 0.42],
    os.path.join('20190522_bsf_e1_n2_train_mass_ss', 'bsf'): [0.44, 0.56, 0.48, 0.48],
    os.path.join('20190522_bsf_e2_n2_train_mass_ss', 'bsf'): [0.6, 0.44, 0.36, 0.56],
    os.path.join('20190525_bsf_ch3_n2_train_inta_ss', 'bsf'): [0.48, 0.56, 0.52, 0.5],
    os.path.join('20190525_bsf_v4_n2_train_mass_ss', 'bsf_1'): [0.46, 0.4, 0.5, 0.46],
    os.path.join('20190527_bsf_v7_k3_n2_train_mass_ss', 'bsf_2'): [0.52, 0.44, 0.48, 0.42],
    os.path.join('20190530_bsf_v10_n2_train_mass_ss', 'bsf'): [0.6, 0.44, 0.56, 0.42],
    os.path.join('20190601_bsf_v11_n2_train_mass_ss', 'filters_32_64_128'): [0.64, 0.36, 0.58, 0.4],
    os.path.join('20190601_bsf_v11_n2_train_mass_ss', 'filters_64_128_256'): [0.62, 0.6, 0.52, 0.44],
    os.path.join('20190603_grid_cwt_fb05_n2_train_mass_ss', 'v12_f_32_64'): [0.66, 0.46, 0.52, 0.46],
    os.path.join('20190605_grid_v15_v16_n2_train_mass_ss', 'v15_timef_64_128_256_cwtf_32_32_fb_0.5'): [0.46, 0.52, 0.62, 0.42],
    os.path.join('20190614_bsf_global_std_n2_train_mass_ss', 'bsf'): [0.62, 0.4, 0.4, 0.48],
    os.path.join('20190617_grid_normalization_n2_train_mass_ss', 'norm_global'): [0.58, 0.42, 0.4, 0.5],
    os.path.join('20190608_bsf_ablation_n2_train_inta_ss', 'v15_tf_64-128-256_cwtf_32-32/rep0'): [0.48, 0.52, 0.5, 0.5],
    os.path.join('20190620_11_12_17_from_scratch_n2_train_dreams_ss', 'v11_None'): [0.22, 0.48, 0.28, 0.38],
    os.path.join('20190620_11_12_17_from_scratch_n2_train_dreams_ss', 'v12_True'): [0.36, 0.5, 0.34, 0.36],
    os.path.join('20190620_11_12_17_from_scratch_n2_train_dreams_ss', 'v17_True'): [0.34, 0.38, 0.28, 0.44],
    os.path.join('20190620_11_12_17_from_scratch_wn_train_dreams_ss', 'v11_None'): [0.36, 0.46, 0.4, 0.36],
    os.path.join('20190620_11_12_17_from_scratch_wn_train_dreams_ss', 'v12_True'): [0.48, 0.46, 0.46, 0.52],
    os.path.join('20190620_11_12_17_from_scratch_wn_train_dreams_ss', 'v17_True'): [0.3, 0.52, 0.28, 0.46],
    os.path.join('20190704_inta_meeting_n2_train_mass_ss', 'v15'): [0.5, 0.42, 0.56, 0.58],
    os.path.join('20190704_inta_meeting_n2_train_mass_ss', 'v20_indep'): [0.4, 0.62, 0.64, 0.56],
    os.path.join('20190704_inta_meeting_n2_train_inta_ss', 'v15'): [0.48, 0.48, 0.5, 0.46],
    os.path.join('20190704_inta_meeting_n2_train_inta_ss', 'v20_indep'): [0.54, 0.42, 0.5, 0.5],
    os.path.join('20190706_inta_05_n2_train_inta_ss', 'v15'): [0.42, 0.44, 0.48, 0.5],
    os.path.join('20190708_grid_v19_pte2_n2_train_mass_ss', 'r_1_i_1_m_1_p_0_fb_0.5'): [0.42, 0.58, 0.62, 0.54],
    os.path.join('20190825_v22_grid_n2_train_mass_ss', 'r_1_i_1_m_1_p_0_drop_0.3_f_64'): [0.52, 0.48, 0.5, 0.52],
    os.path.join('20190827_thesis_1_bsf_e1_n2_train_mass_ss', 'v19'): [0.54, 0.52, 0.64, 0.54],
    os.path.join('20190827_thesis_1_bsf_e2_n2_train_mass_ss', 'v19'): [0.58, 0.5, 0.34, 0.6],
    os.path.join('20190827_thesis_1_bsf_e1_n2_train_mass_kc', 'v19'): [0.54, 0.64, 0.52, 0.5],
    os.path.join('20190827_thesis_1_bsf_e1_n2_train_inta_ss', 'v19'): [0.42, 0.42, 0.46, 0.44],
    os.path.join('20190915_balancing_drop_n2_train_mass_ss', 'v11'): [0.5, 0.5, 0.5, 0.5],
    os.path.join('20190915_balancing_drop_n2_train_mass_ss', 'v19'): [0.5, 0.5, 0.5, 0.5],
    os.path.join('20190915_balancing_drop_n2_train_mass_kc', 'v11'): [0.5, 0.5, 0.5, 0.5],
    os.path.join('20190915_balancing_drop_n2_train_mass_kc', 'v19'): [0.5, 0.5, 0.5, 0.5],
    os.path.join('20190916_balancing_drop_v2_n2_train_mass_ss', 'v11'): [0.5, 0.5, 0.5, 0.5],
    os.path.join('20190916_balancing_drop_v2_n2_train_mass_ss', 'v19'): [0.5, 0.5, 0.5, 0.5],
    os.path.join('20190916_balancing_weight_n2_train_mass_ss', 'v11'): [0.5, 0.5, 0.5, 0.5],
    os.path.join('20190916_balancing_weight_n2_train_mass_ss', 'v19'): [0.5, 0.5, 0.5, 0.5],
    os.path.join('20190917_out_proba_init_grid_n2_train_mass_ss', 'p_0.01_lr_0.0001'): [0.44, 0.54, 0.56, 0.32],
    os.path.join('20190917_out_proba_init_grid_n2_train_mass_ss', 'p_0.1_lr_0.0001'): [0.24, 0.42, 0.54, 0.48],
    os.path.join('20190917_out_proba_init_equal_n2_train_mass_ss', 'v11'): [0.28, 0.58, 0.56, 0.4],
    os.path.join('20190927_out_proba_cwt_grid_n2_train_mass_ss', 'p_0.5_lr_0.0001'): [0.36, 0.42, 0.62, 0.4],
    os.path.join('20190927_out_proba_cwt_grid_n2_train_mass_ss', 'p_0.01_lr_0.0001'): [0.42, 0.46, 0.54, 0.36],
    os.path.join('20191003_loss_grid_cwt_n2_train_mass_ss', 'v19_p_0.5_focal_loss_gamma_1.0'): [0.42, 0.60, 0.58, 0.44],
    os.path.join('20191003_loss_grid_cwt_n2_train_mass_ss', 'v19_p_0.5_focal_loss_gamma_1.5'): [0.44, 0.50, 0.52, 0.52],
    os.path.join('20191003_loss_grid_cwt_n2_train_mass_ss', 'v19_p_0.5_focal_loss_gamma_2.0'): [0.44, 0.52, 0.54, 0.50],
    os.path.join('20191009_focal_loss_grid_n2_train_mass_ss', 'v19_p_0.5_focal_loss_gamma_2.5'): [0.46, 0.54, 0.54, 0.52],
    os.path.join('20191009_focal_loss_grid_n2_train_mass_ss', 'v19_p_0.5_focal_loss_gamma_3.0'): [0.46, 0.52, 0.50, 0.46],
    os.path.join('20191009_focal_loss_grid_n2_train_mass_ss', 'v19_p_0.5_focal_loss_gamma_3.5'): [0.44, 0.54, 0.50, 0.50],
    os.path.join('20191009_focal_loss_grid_n2_train_mass_ss', 'v19_p_0.5_focal_loss_gamma_4.0'): [0.46, 0.52, 0.52, 0.50],
    os.path.join('20190927_loss_grid_n2_train_mass_ss', 'v11_p_0.5_dice_loss_gamma_None'): [0.50, 0.50, 0.50, 0.50],
    os.path.join('20190927_loss_grid_n2_train_mass_ss', 'v11_p_0.5_focal_loss_gamma_1.0'): [0.48, 0.50, 0.54, 0.50],
    os.path.join('20190927_loss_grid_n2_train_mass_ss', 'v11_p_0.5_focal_loss_gamma_1.5'): [0.42, 0.50, 0.56, 0.44],
    os.path.join('20190927_loss_grid_n2_train_mass_ss', 'v11_p_0.5_focal_loss_gamma_2.0'): [0.46, 0.54, 0.54, 0.46],
    os.path.join('20191009_focal_loss_grid_n2_train_mass_ss', 'v11_p_0.5_focal_loss_gamma_2.5'): [0.44, 0.52, 0.46, 0.44],
    os.path.join('20191009_focal_loss_grid_n2_train_mass_ss', 'v11_p_0.5_focal_loss_gamma_3.0'): [0.48, 0.54, 0.54, 0.50],
    os.path.join('20191009_focal_loss_grid_n2_train_mass_ss', 'v11_p_0.5_focal_loss_gamma_3.5'): [0.48, 0.52, 0.52, 0.50],
    os.path.join('20191009_focal_loss_grid_n2_train_mass_ss', 'v11_p_0.5_focal_loss_gamma_4.0'): [0.46, 0.48, 0.52, 0.48],
    os.path.join('20191013_train_at_128Hz_n2_train_mass_ss', 'v19'): [0.26, 0.54, 0.68, 0.34],
    os.path.join('20191017_elastic_grid_pte2_n2_train_mass_ss', 'v11_alpha_0.25_sigma_0.125_keepbest_True'): [0.54, 0.40, 0.42, 0.48],
    os.path.join('20191017_elastic_grid_pte2_n2_train_mass_ss', 'v11_alpha_0.15_sigma_0.075_keepbest_True'): [0.50, 0.34, 0.36, 0.48],
    os.path.join('20191017_elastic_grid_pte2_n2_train_mass_ss', 'v11_alpha_0.25_sigma_0.100_keepbest_False'): [0.36, 0.48, 0.52, 0.48],
    os.path.join('20191106_forced_sep_n2_train_mass_kc', 'v11_sep_0.10'): [0.62, 0.54, 0.54, 0.44],
    os.path.join('20191106_forced_sep_n2_train_mass_kc', 'v11_sep_0.15'): [0.46, 0.52, 0.52, 0.50],
    os.path.join('20191106_forced_sep_n2_train_mass_kc', 'v11_sep_0.00'): [0.60, 0.50, 0.58, 0.52],
    os.path.join('20191106_forced_sep_n2_train_mass_kc', 'v11_sep_0.20'): [0.50, 0.52, 0.54, 0.52],
    os.path.join('20191106_bsf_update_e1_n2_train_mass_ss', 'v19'): [0.44, 0.56, 0.50, 0.48],
    os.path.join('20191106_bsf_update_e1_n2_train_mass_ss', 'v11'): [0.46, 0.52, 0.52, 0.40],
    os.path.join('20191106_bsf_update_e1_n2_train_mass_kc', 'v19'): [0.48, 0.44, 0.44, 0.46],
    os.path.join('20191106_bsf_update_e1_n2_train_mass_kc', 'v11'): [0.48, 0.60, 0.56, 0.56],
    os.path.join('20191106_bsf_update_e2_n2_train_mass_ss', 'v19'): [0.50, 0.48, 0.50, 0.50],
    os.path.join('20191106_bsf_update_e2_n2_train_mass_ss', 'v11'): [0.60, 0.52, 0.48, 0.54],

    os.path.join('20191227_bsf_10runs_e1_n2_train_mass_ss', 'v19'): [0.44, 0.56, 0.58, 0.46, 0.44, 0.50, 0.58, 0.52, 0.48, 0.50],
    os.path.join('20191227_bsf_10runs_e1_n2_train_mass_ss', 'v11'): [0.44, 0.50, 0.54, 0.46, 0.40, 0.50, 0.52, 0.48, 0.48, 0.50],
    os.path.join('20191227_bsf_10runs_e2_n2_train_mass_ss', 'v19'): [0.50, 0.54, 0.48, 0.48, 0.48, 0.46, 0.54, 0.44, 0.52, 0.48],
    os.path.join('20191227_bsf_10runs_e2_n2_train_mass_ss', 'v11'): [0.50, 0.52, 0.46, 0.48, 0.48, 0.54, 0.54, 0.48, 0.52, 0.48],
    os.path.join('20191227_bsf_10runs_e1_n2_train_mass_kc', 'v19'): [0.44, 0.46, 0.48, 0.48, 0.42, 0.54, 0.44, 0.40, 0.50, 0.44],
    os.path.join('20191227_bsf_10runs_e1_n2_train_mass_kc', 'v11'): [0.60, 0.60, 0.58, 0.58, 0.56, 0.54, 0.58, 0.56, 0.42, 0.60],

    os.path.join('20200407_attention_grid_n2_train_mass_ss', 'att01_d_512_h_8_fc_0'): [0.36, 0.50, 0.54, 0.46],
    os.path.join('20200407_attention_grid_n2_train_mass_ss', 'att01_d_512_h_8_fc_128'): [0.44, 0.50, 0.48, 0.46],
    os.path.join('20200407_attention_grid_n2_train_mass_ss', 'att01_d_256_h_8_fc_128'): [0.42, 0.46, 0.48, 0.46],
    os.path.join('20200407_attention_grid_n2_train_mass_ss', 'att01_d_256_h_8_fc_0'): [0.42, 0.42, 0.50, 0.44],

    os.path.join('20200408_att03_grid_n2_train_mass_ss', 'att03_d_256_h_8_fc_0'): [0.40, 0.54, 0.54, 0.44],
    os.path.join('20200408_att03_grid_n2_train_mass_ss', 'att03_d_512_h_8_fc_0'): [0.44, 0.52, 0.50, 0.40],
    os.path.join('20200408_att03_grid_n2_train_mass_ss', 'att03_d_512_h_8_fc_128'): [0.46, 0.50, 0.52, 0.44],
    os.path.join('20200408_att03_grid_n2_train_mass_ss', 'att03_d_256_h_8_fc_128'): [0.46, 0.50, 0.48, 0.44],

    os.path.join('20200409_attention_grid_n2_train_mass_kc', 'att01_d_512_h_8_fc_0'): [0.30, 0.34, 0.32, 0.30],
    os.path.join('20200409_attention_grid_n2_train_mass_kc', 'att01_d_512_h_8_fc_128'): [0.38, 0.36, 0.36, 0.40],
    os.path.join('20200409_attention_grid_n2_train_mass_kc', 'att01_d_256_h_8_fc_0'): [0.36, 0.32, 0.30, 0.28],
    os.path.join('20200409_attention_grid_n2_train_mass_kc', 'att01_d_256_h_8_fc_128'): [0.36, 0.36, 0.36, 0.40],

    os.path.join('20200409_att03_grid_n2_train_mass_kc', 'att03_d_512_h_8_fc_0'): [0.40, 0.42, 0.36, 0.38],
    os.path.join('20200409_att03_grid_n2_train_mass_kc', 'att03_d_256_h_8_fc_128'): [0.42, 0.44, 0.46, 0.40],
    os.path.join('20200409_att03_grid_n2_train_mass_kc', 'att03_d_256_h_8_fc_0'): [0.38, 0.42, 0.38, 0.42],
    os.path.join('20200409_att03_grid_n2_train_mass_kc', 'att03_d_512_h_8_fc_128'): [0.40, 0.46, 0.40, 0.42] 

}
