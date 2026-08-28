from automl.prepare_run import normalize_feature_name


def test_normalize_feature_name_strips_csv_and_json_prefixes():
    assert normalize_feature_name("feature_fv_charge") == "fv_charge"
    assert normalize_feature_name("surface_hydrophobic_patch") == "hydrophobic_patch"
    assert normalize_feature_name("general_cdr_h3_length") == "cdr_h3_length"
    assert normalize_feature_name("sequence_motives_n_glycosylation") == "n_glycosylation"
    assert normalize_feature_name("core_packing") == "core_packing"
    assert normalize_feature_name("  fv_csp  ") == "fv_csp"
