"""Restored original TARF x post-inversion physical residual Transformer."""
SEEDS = (1107, 1108, 1109)
# A/R/S/H = TARF / Transformer / SAB / GLIN.
CODES = ('0001', '1001', '0101', '1101')
TRAIN_CODES = ('0001C', '1001C', '0101', '1101')
CONTINUATION_CODES = ('0001C', '1001C')
SPECIAL_CODES = ()
ALL_CODES = CODES + CONTINUATION_CODES
ALL = ALL_CODES
LABELS = {'0001': 'Base', '1001': 'Base-TARF',
          '0101': 'Base-Correction', '1101': 'Full (TARF+Correction)',
          '0001C': 'Base-CT', '1001C': 'Base-TARF-CT'}
DISPLAY_ORDER = tuple(LABELS)
REFERENCE_CODES = CODES
REFINER = dict(width=16, dim=32, heads=4, blocks=1, correction_bound=.25,
               density_scale=.85, sensitivity_floor=1e-3, support_gated=True)
ACCEPTANCE = dict(single_sae_reduction_min=.20, full_best_single_sae_reduction_min=.05,
                  historical_single_regression_max=.02, physics_regression_max=.02,
                  dice_drop_max=.002, pdacc_drop_max=.01)


def comparison_plan():
    """Single-seed descriptive SAE reductions; not significance tests."""
    return [
        dict(id='TARF_vs_Base', coefficients={'0001': 1., '1001': -1.}),
        dict(id='ResidualTransformer_vs_Base', coefficients={'0001': 1., '0101': -1.}),
        dict(id='TARF_given_ResidualTransformer', coefficients={'0101': 1., '1101': -1.}),
        dict(id='ResidualTransformer_given_TARF', coefficients={'1001': 1., '1101': -1.}),
        dict(id='Full_vs_Base', coefficients={'0001': 1., '1101': -1.}),
        dict(id='additive_interaction',
             coefficients={'1001': 1., '0101': 1., '0001': -1., '1101': -1.}),
    ]
