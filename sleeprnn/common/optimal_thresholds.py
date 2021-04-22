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
    os.path.join('20191227_bsf_10runs_e1_n2_train_mass_kc', 'v19'): [0.44, 0.46, 0.48, 0.48, 0.44, 0.54, 0.44, 0.44, 0.50, 0.44],
    os.path.join('20191227_bsf_10runs_e1_n2_train_mass_kc', 'v11'): [0.60, 0.60, 0.58, 0.58, 0.56, 0.54, 0.58, 0.56, 0.52, 0.60],
    os.path.join('20191227_bsf_10runs_e1_n2_train_inta_ss', 'v19'): [0.50, 0.44, 0.46, 0.50, 0.52, 0.50, 0.46, 0.50, 0.48, 0.46],
    os.path.join('20191227_bsf_10runs_e1_n2_train_inta_ss', 'v11'): [0.48, 0.44, 0.46, 0.46, 0.48, 0.52, 0.48, 0.46, 0.42, 0.42],

    os.path.join('20200407_attention_grid_n2_train_mass_ss', 'att01_d_512_h_8_fc_0'): [0.38, 0.50, 0.54, 0.46],
    os.path.join('20200407_attention_grid_n2_train_mass_ss', 'att01_d_512_h_8_fc_128'): [0.44, 0.50, 0.48, 0.46],
    os.path.join('20200407_attention_grid_n2_train_mass_ss', 'att01_d_256_h_8_fc_128'): [0.42, 0.46, 0.50, 0.46],
    os.path.join('20200407_attention_grid_n2_train_mass_ss', 'att01_d_256_h_8_fc_0'): [0.42, 0.42, 0.50, 0.44],

    os.path.join('20200408_att03_grid_n2_train_mass_ss', 'att03_d_256_h_8_fc_0'): [0.40, 0.54, 0.54, 0.44],
    os.path.join('20200408_att03_grid_n2_train_mass_ss', 'att03_d_512_h_8_fc_0'): [0.44, 0.52, 0.50, 0.40],
    os.path.join('20200408_att03_grid_n2_train_mass_ss', 'att03_d_512_h_8_fc_128'): [0.46, 0.50, 0.52, 0.44],
    os.path.join('20200408_att03_grid_n2_train_mass_ss', 'att03_d_256_h_8_fc_128'): [0.46, 0.50, 0.48, 0.44],

    os.path.join('20200409_attention_grid_n2_train_mass_kc', 'att01_d_512_h_8_fc_0'): [0.32, 0.34, 0.32, 0.30],
    os.path.join('20200409_attention_grid_n2_train_mass_kc', 'att01_d_512_h_8_fc_128'): [0.38, 0.36, 0.36, 0.40],
    os.path.join('20200409_attention_grid_n2_train_mass_kc', 'att01_d_256_h_8_fc_0'): [0.36, 0.32, 0.30, 0.28],
    os.path.join('20200409_attention_grid_n2_train_mass_kc', 'att01_d_256_h_8_fc_128'): [0.38, 0.36, 0.36, 0.40],

    os.path.join('20200409_att03_grid_n2_train_mass_kc', 'att03_d_512_h_8_fc_0'): [0.40, 0.42, 0.36, 0.38],
    os.path.join('20200409_att03_grid_n2_train_mass_kc', 'att03_d_512_h_8_fc_128'): [0.40, 0.46, 0.48, 0.42],
    os.path.join('20200409_att03_grid_n2_train_mass_kc', 'att03_d_256_h_8_fc_128'): [0.42, 0.44, 0.46, 0.40],
    os.path.join('20200409_att03_grid_n2_train_mass_kc', 'att03_d_256_h_8_fc_0'): [0.44, 0.42, 0.38, 0.42],

    os.path.join('20200409_att04_head_grid_n2_train_mass_ss', 'att04_h_08'): [0.46, 0.52, 0.52, 0.44],
    os.path.join('20200410_att04_task_pe_grid_n2_train_mass_ss', 'att04_pe_10000'): [0.44, 0.56, 0.50, 0.50],
    os.path.join('20200410_att04_task_pe_grid_n2_train_mass_kc', 'att04_pe_10000'): [0.52, 0.52, 0.58, 0.58],

    os.path.join('20200502_timePLUScwt_fb_10runs_e1_n2_train_mass_ss', 'v35_fb_1.0'): [0.42, 0.52, 0.50, 0.44, 0.46, 0.48, 0.50, 0.46, 0.48, 0.50],
    os.path.join('20200502_timePLUScwt_fb_10runs_e1_n2_train_mass_ss', 'v35_fb_0.5'): [0.44, 0.48, 0.50, 0.44, 0.44, 0.48, 0.50, 0.46, 0.46, 0.46],
    os.path.join('20200502_timePLUScwt_fb_10runs_e1_n2_train_mass_ss', 'v35_fb_1.5'): [0.44, 0.54, 0.54, 0.42, 0.50, 0.48, 0.50, 0.48, 0.44, 0.48],

    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_cross_entropy_smoothing_clip_loss_bNone_e0.05_gNone_pi0.05'): [0.36, 0.44],
    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_cross_entropy_smoothing_clip_loss_bNone_e0.1_gNone_pi0.1'): [0.42, 0.46],
    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_cross_entropy_loss_bNone_eNone_gNone_pi0.01'): [0.32, 0.46],
    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_cross_entropy_smoothing_loss_bNone_e0.1_gNone_pi0.1'): [0.42, 0.48],
    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_cross_entropy_neg_entropy_loss_b0.5_eNone_gNone_pi0.1'): [0.44, 0.48],
    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_cross_entropy_hard_clip_loss_bNone_e0.1_gNone_pi0.1'): [0.42, 0.44],
    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_cross_entropy_neg_entropy_loss_b0.4_eNone_gNone_pi0.05'): [0.40, 0.50],
    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_focal_loss_bNone_eNone_g1.5_pi0.1'): [0.36, 0.44],
    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_cross_entropy_hard_clip_loss_bNone_e0.05_gNone_pi0.05'): [0.40, 0.42],
    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_focal_loss_bNone_eNone_g1.0_pi0.05'): [0.38, 0.44],
    os.path.join('20200502_grid_losses_2020_n2_train_mass_ss', 'v11_cross_entropy_smoothing_loss_bNone_e0.05_gNone_pi0.05'): [0.40, 0.44],

    os.path.join('20200522_normalizations_n2_train_mass_ss', 'v11_norm_global'): [0.40, 0.50, 0.50, 0.44],
    os.path.join('20200522_normalizations_n2_train_mass_ss', 'v11_norm_iqr'): [0.40, 0.54, 0.48, 0.42],
    os.path.join('20200522_normalizations_n2_train_mass_ss', 'v11_norm_std'): [0.42, 0.52, 0.50, 0.50],

    os.path.join('20200529_worst_negatives_2020_v2_n2_train_mass_ss', 'v11_r4_min100'): [0.40, 0.40, 0.50, 0.36],

    os.path.join('20200530_border_weights_n2_train_mass_ss', 'v11_xent_borders_ind_a04_hw08'): [0.48, 0.62, 0.56, 0.50],
    os.path.join('20200529_mod_focal_ind_n2_train_mass_ss', 'v11_g3_w02_pi0.10'): [0.38, 0.40, 0.44, 0.38],
    os.path.join('20200602_best_border_weights_n2_train_mass_kc', 'v11_xent_borders_ind_a04_hw08'): [0.48, 0.60, 0.58, 0.62],
    os.path.join('20200602_best_mod_focal_ind_n2_train_mass_kc', 'v11_g3_w02_pi0.10'): [0.46, 0.46, 0.52, 0.46],

    os.path.join('20200603_best_mix_weights_n2_train_mass_ss', 'v11_a4_m2_c0.25_sum'): [0.34, 0.38, 0.42, 0.36],
    os.path.join('20200603_best_mix_weights_n2_train_mass_ss', 'v11_a4_m2_c0.50_sum'): [0.40, 0.44, 0.46, 0.38],
    os.path.join('20200603_best_mix_weights_n2_train_mass_ss', 'v11_a4_m2_c1.00_sum'): [0.44, 0.48, 0.50, 0.46],
    os.path.join('20200603_best_mix_weights_n2_train_mass_ss', 'v11_a4_m2_c2.00_sum'): [0.48, 0.54, 0.52, 0.46],

    os.path.join('20200603_best_mix_weights_n2_train_mass_kc', 'v11_a4_m2_c2.00_sum'): [0.60, 0.60, 0.56, 0.52],
    os.path.join('20200603_best_mix_weights_n2_train_mass_kc', 'v11_a4_m2_c1.00_sum'): [0.46, 0.46, 0.48, 0.46],
    os.path.join('20200603_best_mix_weights_n2_train_mass_kc', 'v11_a4_m2_c0.50_sum'): [0.42, 0.38, 0.44, 0.38],
    os.path.join('20200603_best_mix_weights_n2_train_mass_kc', 'v11_a4_m2_c0.25_sum'): [0.34, 0.34, 0.42, 0.38],

    os.path.join('20200604_alt_mix_weights_n2_train_mass_ss', 'v11_c0.25_regular_xent'): [0.40, 0.48, 0.44, 0.40],
    os.path.join('20200604_alt_mix_weights_n2_train_mass_ss', 'v11_c0.25_softclip_xent'): [0.38, 0.42, 0.46, 0.40],

    os.path.join('20200606_var_reg_first_grid_n2_train_mass_ss', 'v11_reg_-5.0'): [0.34, 0.42],
    os.path.join('20200606_var_reg_first_grid_n2_train_mass_ss', 'v11_reg_-1.0'): [0.36, 0.40],
    os.path.join('20200606_var_reg_first_grid_n2_train_mass_ss', 'v11_reg_-3.0'): [0.36, 0.40],
    os.path.join('20200606_var_reg_first_grid_n2_train_mass_ss', 'v11_reg_-6.0'): [0.36, 0.42],
    os.path.join('20200606_var_reg_first_grid_n2_train_mass_ss', 'v11_reg_1.0'): [0.34, 0.38],
    os.path.join('20200606_var_reg_first_grid_n2_train_mass_ss', 'v11_reg_-2.0'): [0.32, 0.40],
    os.path.join('20200606_var_reg_first_grid_n2_train_mass_ss', 'v11_reg_0.0'): [0.34, 0.42],
    os.path.join('20200606_var_reg_first_grid_n2_train_mass_ss', 'v11_reg_-4.0'): [0.32, 0.40],

    os.path.join('20200607_var_reg_bimodal_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v4_0.0'): [0.32, 0.38],
    os.path.join('20200607_var_reg_bimodal_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v4_-1.0'): [0.34, 0.40],
    os.path.join('20200607_var_reg_bimodal_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v3_-2.0'): [0.34, 0.42],
    os.path.join('20200607_var_reg_bimodal_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v4_-2.0'): [0.36, 0.40],
    os.path.join('20200607_var_reg_bimodal_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v3_-1.0'): [0.36, 0.38],
    os.path.join('20200607_var_reg_bimodal_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v3_-3.0'): [0.34, 0.38],
    os.path.join('20200607_var_reg_bimodal_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v3_0.0'): [0.36, 0.42],
    os.path.join('20200607_var_reg_bimodal_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v4_-3.0'): [0.32, 0.40],

    os.path.join('20200610_hinge_loss_n2_train_mass_ss', 'v11_hinge_loss_pi0.5'): [0.30, 0.38, 0.44, 0.30],
    os.path.join('20200610_hinge_loss_n2_train_mass_kc', 'v11_hinge_loss_pi0.5'): [0.50, 0.60, 0.56, 0.52],

    os.path.join('20200610_var_reg_lagged_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v3_lag4_-1.0'): [0.32, 0.40],
    os.path.join('20200610_var_reg_lagged_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v4_lag4_-1.0'): [0.34, 0.40],
    os.path.join('20200610_var_reg_lagged_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v4_lag4_-2.0'): [0.38, 0.40],
    os.path.join('20200610_var_reg_lagged_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v3_lag4_-4.0'): [0.36, 0.42],
    os.path.join('20200610_var_reg_lagged_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v4_lag4_-3.0'): [0.32, 0.38],
    os.path.join('20200610_var_reg_lagged_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v3_lag4_-2.0'): [0.34, 0.40],
    os.path.join('20200610_var_reg_lagged_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v4_lag4_-4.0'): [0.34, 0.42],
    os.path.join('20200610_var_reg_lagged_first_grid_n2_train_mass_ss', 'v11_weighted_cross_entropy_loss_v3_lag4_-3.0'): [0.36, 0.42],

    os.path.join('20200724_reproduce_red_n2_train_mass_ss', 'v19_rep2'): [0.46, 0.54, 0.56, 0.40],
    os.path.join('20200724_reproduce_red_n2_train_mass_ss', 'v19_rep4'): [0.44, 0.52, 0.60, 0.46],
    os.path.join('20200724_reproduce_red_n2_train_mass_ss', 'v19_rep1'): [0.52, 0.56, 0.56, 0.48],
    os.path.join('20200724_reproduce_red_n2_train_mass_ss', 'v19_rep3'): [0.42, 0.52, 0.50, 0.46],
    os.path.join('20200724_reproduce_red_n2_train_mass_ss', 'v11_rep2'): [0.44, 0.52, 0.42, 0.38],
    os.path.join('20200724_reproduce_red_n2_train_mass_ss', 'v11_rep3'): [0.42, 0.48, 0.44, 0.44],
    os.path.join('20200724_reproduce_red_n2_train_mass_ss', 'v11_rep4'): [0.42, 0.52, 0.46, 0.42],
    os.path.join('20200724_reproduce_red_n2_train_mass_ss', 'v11_rep1'): [0.48, 0.48, 0.52, 0.44],

    os.path.join('20200724_reproduce_red_n2_train_mass_kc', 'v19_rep3'): [0.42, 0.48, 0.48, 0.42],
    os.path.join('20200724_reproduce_red_n2_train_mass_kc', 'v19_rep2'): [0.52, 0.50, 0.42, 0.48],
    os.path.join('20200724_reproduce_red_n2_train_mass_kc', 'v19_rep1'): [0.50, 0.52, 0.50, 0.52],
    os.path.join('20200724_reproduce_red_n2_train_mass_kc', 'v19_rep4'): [0.56, 0.54, 0.46, 0.38],
    os.path.join('20200724_reproduce_red_n2_train_mass_kc', 'v11_rep3'): [0.60, 0.58, 0.56, 0.46],
    os.path.join('20200724_reproduce_red_n2_train_mass_kc', 'v11_rep2'): [0.60, 0.52, 0.56, 0.50],
    os.path.join('20200724_reproduce_red_n2_train_mass_kc', 'v11_rep4'): [0.50, 0.58, 0.50, 0.58],
    os.path.join('20200724_reproduce_red_n2_train_mass_kc', 'v11_rep1'): [0.52, 0.56, 0.46, 0.56],

    os.path.join('20200914_fft_based_norm_n2_train_mass_ss', 'v11_rep2'): [0.40, 0.56, 0.52, 0.46],
    os.path.join('20200914_fft_based_norm_n2_train_mass_ss', 'v11_rep1'): [0.50, 0.54, 0.54, 0.42],
    os.path.join('20200914_fft_based_norm_n2_train_mass_ss', 'v19_rep2'): [0.46, 0.58, 0.54, 0.46],
    os.path.join('20200914_fft_based_norm_n2_train_mass_ss', 'v19_rep1'): [0.44, 0.52, 0.62, 0.48],

    os.path.join('20210211_cap_init_check_n2_train_cap_ss', 'v11_mkd2'): [0.44, 0.36, 0.42, 0.36],

    os.path.join('20210212_cap_long_training_n2_train_cap_ss', 'v11_mkd2'): [0.58, 0.48, 0.44, 0.58],

    os.path.join('20210216_cap41_n2_train_cap_all_ss', 'v11_mkd2'): [0.38, 0.46, 0.42, 0.42, 0.38, 0.40, 0.44],

    os.path.join('20210218_cap_focal_n2_train_cap_all_ss', 'v11_mkd2'): [0.32, 0.36, 0.34, 0.30, 0.32, 0.32, 0.32],
    
    os.path.join('20210219_wave_augment_n2_train_cap_all_ss', 'v11_mkd2'): [0.38, 0.52, 0.40, 0.40, 0.40, 0.40, 0.46],
    
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.1_decay0.5_loss-xentropy_from-xentropy'): [0.42, 0.52, 0.50, 0.36],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init1.0_decay0.1_loss-xentropy_from-xentropy'): [0.40, 0.46, 0.50, 0.40],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.5_decay0.5_loss-focal_from-xentropy'): [0.44, 0.46, 0.48, 0.42],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.1_decay0.5_loss-xentropy_from-focal'): [0.36, 0.46, 0.46, 0.38],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init1.0_decay0.1_loss-xentropy_from-focal'): [0.40, 0.44, 0.52, 0.40],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.5_decay0.5_loss-xentropy_from-focal'): [0.42, 0.46, 0.48, 0.36],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.5_decay0.5_loss-xentropy_from-xentropy'): [0.48, 0.42, 0.48, 0.36],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init1.0_decay0.5_loss-focal_from-xentropy'): [0.42, 0.48, 0.48, 0.46],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init1.0_decay0.1_loss-focal_from-xentropy'): [0.44, 0.46, 0.48, 0.44],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init1.0_decay0.1_loss-focal_from-focal'): [0.42, 0.48, 0.48, 0.42],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init1.0_decay0.5_loss-xentropy_from-focal'): [0.40, 0.50, 0.50, 0.44],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init1.0_decay0.5_loss-xentropy_from-xentropy'): [0.44, 0.50, 0.46, 0.38],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.5_decay0.1_loss-xentropy_from-xentropy'): [0.38, 0.48, 0.46, 0.36],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.5_decay0.5_loss-focal_from-focal'): [0.42, 0.46, 0.48, 0.44],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init1.0_decay0.5_loss-focal_from-focal'): [0.44, 0.48, 0.52, 0.42],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.1_decay0.5_loss-focal_from-xentropy'): [0.46, 0.46, 0.50, 0.42],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.5_decay0.1_loss-focal_from-focal'): [0.42, 0.50, 0.46, 0.44],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.1_decay0.5_loss-focal_from-focal'): [0.44, 0.50, 0.46, 0.44],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.5_decay0.1_loss-focal_from-xentropy'): [0.42, 0.48, 0.48, 0.44],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.5_decay0.1_loss-xentropy_from-focal'): [0.38, 0.46, 0.44, 0.40],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.1_decay0.1_loss-xentropy_from-focal'): [0.40, 0.44, 0.48, 0.40],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.1_decay0.1_loss-xentropy_from-xentropy'): [0.38, 0.44, 0.44, 0.40],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.1_decay0.1_loss-focal_from-focal'): [0.46, 0.44, 0.48, 0.44],
    os.path.join('20210221_mvp_expert_finetuning_n2_train_mass_ss', 'v11_mkd2_init0.1_decay0.1_loss-focal_from-xentropy'): [0.42, 0.46, 0.46, 0.42],
    
    os.path.join('20210317_annot_effect_4cv_exp1_n2_train_cap_full_ss', 'v11'): [0.32, 0.44, 0.42, 0.36],
    os.path.join('20210318_annot_effect_4cv_exp2_n2_train_cap_full_ss', 'v11'): [0.42, 0.44, 0.46, 0.34],

    os.path.join('20210318_cap80_finetuning_n2_train_mass_ss', 'v11_epochs40_from-ckpt1'): [0.48, 0.56, 0.42, 0.32],
    os.path.join('20210318_cap80_finetuning_n2_train_mass_ss', 'v11_epochs20_from-ckpt2'): [0.38, 0.46, 0.58, 0.42],
    os.path.join('20210318_cap80_finetuning_n2_train_mass_ss', 'v11_epochs40_from-ckpt2'): [0.40, 0.54, 0.56, 0.44],
    os.path.join('20210318_cap80_finetuning_n2_train_mass_ss', 'v11_epochs10_from-ckpt2'): [0.34, 0.50, 0.50, 0.40],
    os.path.join('20210318_cap80_finetuning_n2_train_mass_ss', 'v11_epochs20_from-ckpt1'): [0.42, 0.50, 0.32, 0.42],
    os.path.join('20210318_cap80_finetuning_n2_train_mass_ss', 'v11_epochs10_from-ckpt1'): [0.40, 0.44, 0.44, 0.42],
    os.path.join('20210318_cap80_finetuning_n2_train_mass_ss', 'v11_epochs5_from-ckpt2'): [0.36, 0.42, 0.42, 0.46],
    os.path.join('20210318_cap80_finetuning_n2_train_mass_ss', 'v11_epochs5_from-ckpt1'): [0.30, 0.44, 0.48, 0.32],
    os.path.join('20210318_cap80_finetuning_n2_train_mass_ss', 'v11_epochs1_from-ckpt2'): [0.32, 0.40, 0.42, 0.32],
    os.path.join('20210318_cap80_finetuning_n2_train_mass_ss', 'v11_epochs1_from-ckpt1'): [0.20, 0.34, 0.30, 0.26],
    
    os.path.join('20210323_annot_effect_4cv_exp1_n2_train_cap_full_ss', 'v19'): [0.46, 0.38, 0.44, 0.40],
    os.path.join('20210323_annot_effect_4cv_exp2_n2_train_cap_full_ss', 'v19'): [0.42, 0.48, 0.48, 0.36],
    
    os.path.join('20210323_cap80_finetuning_n2_train_mass_ss', 'v19_epochs20_from-ckpt2'): [0.36, 0.48, 0.56, 0.40],
    os.path.join('20210323_cap80_finetuning_n2_train_mass_ss', 'v19_epochs20_from-ckpt1'): [0.38, 0.54, 0.50, 0.52],
    os.path.join('20210323_cap80_finetuning_n2_train_mass_ss', 'v19_epochs40_from-ckpt1'): [0.48, 0.48, 0.56, 0.32],
    os.path.join('20210323_cap80_finetuning_n2_train_mass_ss', 'v19_epochs10_from-ckpt1'): [0.38, 0.52, 0.50, 0.48],
    os.path.join('20210323_cap80_finetuning_n2_train_mass_ss', 'v19_epochs40_from-ckpt2'): [0.52, 0.54, 0.34, 0.48],
    os.path.join('20210323_cap80_finetuning_n2_train_mass_ss', 'v19_epochs10_from-ckpt2'): [0.38, 0.48, 0.42, 0.48],
    os.path.join('20210323_cap80_finetuning_n2_train_mass_ss', 'v19_epochs5_from-ckpt2'): [0.34, 0.48, 0.54, 0.48],
    os.path.join('20210323_cap80_finetuning_n2_train_mass_ss', 'v19_epochs5_from-ckpt1'): [0.42, 0.42, 0.54, 0.36],
    os.path.join('20210323_cap80_finetuning_n2_train_mass_ss', 'v19_epochs1_from-ckpt2'): [0.32, 0.42, 0.44, 0.36],
    os.path.join('20210323_cap80_finetuning_n2_train_mass_ss', 'v19_epochs1_from-ckpt1'): [0.22, 0.30, 0.26, 0.22],

    os.path.join('20210324_swish_n2_train_mass_ss', 'v11_mkd2_swish'): [0.34, 0.46, 0.46, 0.34],
    
    os.path.join('20210324_annot_effect_4cv_exp1_n2_train_cap_full_ss', 'v11_mkd2'): [0.34, 0.40, 0.38, 0.32],
    os.path.join('20210325_annot_effect_4cv_exp2_n2_train_cap_full_ss', 'v11_mkd2'): [0.44, 0.46, 0.38, 0.34],
    
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f64_02-04-00_d8'): [0.28, 0.28, 0.38, 0.34],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f64_02-03-00_d8'): [0.30, 0.32, 0.34, 0.36],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-03-03_d4'): [0.28, 0.34, 0.30, 0.28],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-06-03_d4'): [0.26, 0.32, 0.32, 0.30],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-06-09_d1'): [0.32, 0.30, 0.34, 0.32],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-03-06_d1'): [0.30, 0.40, 0.40, 0.30],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f64_02-05-00_d8'): [0.24, 0.42, 0.30, 0.36],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-03-04_d4'): [0.26, 0.34, 0.36, 0.34],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-03-05_d4'): [0.32, 0.38, 0.34, 0.26],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-06-06_d1'): [0.36, 0.38, 0.30, 0.38],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-06-04_d4'): [0.30, 0.26, 0.30, 0.30],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-06-03_d1'): [0.36, 0.42, 0.28, 0.26],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-03-03_d1'): [0.34, 0.44, 0.32, 0.30],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-03-09_d1'): [0.32, 0.34, 0.32, 0.32],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f32_02-06-05_d4'): [0.32, 0.36, 0.28, 0.32],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f64_02-06-00_d1'): [0.40, 0.42, 0.38, 0.36],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f64_02-09-00_d1'): [0.32, 0.32, 0.38, 0.42],
    os.path.join('20210326_biggernet_gridconv_exp1_n2_train_cap_full_ss', 'v41_f64_02-03-00_d1'): [0.32, 0.36, 0.32, 0.36],
    
    os.path.join('20210330_biggernet_lstm_exp1_n2_train_cap_full_ss', 'v41_256-256'): [0.30, 0.36, 0.34, 0.32],
    os.path.join('20210330_biggernet_lstm_exp1_n2_train_cap_full_ss', 'v41_256-128'): [0.38, 0.38, 0.36, 0.34],
    os.path.join('20210330_biggernet_lstm_exp1_n2_train_cap_full_ss', 'v41_128-128'): [0.36, 0.34, 0.34, 0.32],
    os.path.join('20210330_biggernet_lstm_exp1_n2_train_cap_full_ss', 'v41_256-000'): [0.30, 0.36, 0.26, 0.26],
    os.path.join('20210330_biggernet_lstm_exp1_n2_train_cap_full_ss', 'v41_128-000'): [0.34, 0.36, 0.26, 0.30],
    os.path.join('20210330_biggernet_lstm_exp1_n2_train_cap_full_ss', 'v41_128-064'): [0.30, 0.32, 0.30, 0.32],

    os.path.join('20210330_biggernet_att_exp1_n2_train_cap_full_ss', 'v42_n2_layernorm'): [0.34, 0.42, 0.34, 0.36],
    os.path.join('20210330_biggernet_att_exp1_n2_train_cap_full_ss', 'v42_n4_layernorm'): [0.26, 0.40, 0.38, 0.32],
    os.path.join('20210330_biggernet_att_exp1_n2_train_cap_full_ss', 'v42_n2_batchnorm'): [0.38, 0.42, 0.36, 0.36],
    os.path.join('20210330_biggernet_att_exp1_n2_train_cap_full_ss', 'v42_n4_batchnorm'): [0.32, 0.36, 0.30, 0.24],

    os.path.join('20210405_check_contralateral_exp1_n2_train_mass_ss_alt', 'v11_mkd2'): [0.50, 0.50, 0.50, 0.42],
    
    os.path.join('20210409_page_mask_exp1_n2_train_mass_ss', 'v11_mkd2'): [0.46, 0.50, 0.50, 0.46],
    
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_lstm_from-none'): [0.48, 0.50, 0.52, 0.40],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_res-lstm-ln_from-ckpt2'): [0.40, 0.50, 0.42, 0.40],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_res-lstm-ln_from-ckpt1'): [0.48, 0.50, 0.46, 0.42],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_res-lstm-bn_from-ckpt1'): [0.40, 0.50, 0.48, 0.40],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_lstm_from-ckpt1'): [0.36, 0.46, 0.42, 0.36],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_lstm_from-ckpt2'): [0.44, 0.44, 0.48, 0.38],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_res-lstm-bn_from-ckpt2'): [0.40, 0.46, 0.46, 0.36],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_res-lstm-ln_from-none'): [0.46, 0.52, 0.46, 0.44],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_res-lstm-bn_from-none'): [0.38, 0.52, 0.50, 0.40],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_att2_from-ckpt1'): [0.36, 0.52, 0.50, 0.36],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_att3_from-ckpt2'): [0.40, 0.46, 0.46, 0.34],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_att3_from-ckpt1'): [0.32, 0.48, 0.48, 0.36],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_att2_from-ckpt2'): [0.48, 0.40, 0.52, 0.38],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_att3_from-none'): [0.40, 0.52, 0.52, 0.36],
    os.path.join('20210410_lego_2_finetuning_n2_train_mass_ss', 'v43_res-multi-d8_att2_from-none'): [0.34, 0.50, 0.54, 0.42],

    os.path.join('20210411_bsf_update_exp1_n2_train_mass_ss', 'v43'): [0.42, 0.52, 0.54, 0.42],
    os.path.join('20210411_bsf_update_exp2_n2_train_mass_ss', 'v43'): [0.48, 0.52, 0.54, 0.44],

    os.path.join('20210411_5fold-cv_exp1_n2_train_moda_ss', 'v43'): [0.50, 0.54, 0.54, 0.52, 0.50],
    os.path.join('20210413_10fold-cv_exp1_n2_train_moda_ss', 'v43'): [0.50, 0.48, 0.48, 0.54, 0.48, 0.46, 0.52, 0.48, 0.62, 0.50],
    os.path.join('20210415_5fold-cv_exp1_n2_train_inta_ss', 'v43'): [0.42, 0.48, 0.46, 0.46, 0.40],
    
    os.path.join('20210420_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best1_reset1_metric_criterion'): [0.44, 0.48, 0.44, 0.52, 0.44],
    os.path.join('20210420_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best0_reset0_metric_criterion'): [0.46, 0.56, 0.44, 0.46, 0.48],
    os.path.join('20210420_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best0_reset1_loss_criterion'): [0.48, 0.48, 0.46, 0.52, 0.42],
    os.path.join('20210420_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best0_reset1_metric_criterion'): [0.52, 0.58, 0.40, 0.52, 0.44],
    os.path.join('20210420_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best0_reset0_loss_criterion'): [0.48, 0.52, 0.40, 0.52, 0.48],
    os.path.join('20210420_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best1_reset1_loss_criterion'): [0.44, 0.36, 0.54, 0.36, 0.56],
    os.path.join('20210420_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best1_reset0_metric_criterion'): [0.52, 0.46, 0.36, 0.56, 0.50],
    os.path.join('20210420_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best1_reset0_loss_criterion'): [0.46, 0.22, 0.50, 0.30, 0.48],
    
    os.path.join('20210421_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best1_reset0_metric_criterion'): [0.44, 0.42, 0.46, 0.54, 0.44],
    os.path.join('20210421_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best1_reset0_loss_criterion'): [0.40, 0.52, 0.54, 0.36, 0.56],
    os.path.join('20210421_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best0_reset0_loss_criterion'): [0.48, 0.46, 0.54, 0.50, 0.44],
    os.path.join('20210421_val_usage_5fold-cv_exp1_n2_train_mass_ss', 'v43_best0_reset0_metric_criterion'): [0.54, 0.54, 0.40, 0.46, 0.44],
}



