"""Normalisation choices for the scatter-normalisation tutorial."""

from simind_python_connector.recon import measured_normaliser, reference_normaliser


STRATEGIES = (
    "Known activity",
    "Reference, sum",
    "Reference, trimmed",
    "Dewaraja, global",
    "Dewaraja, pixel-wise",
)


def strategy_options(full_model, measured, activity_mbq, time_per_projection_s):
    """Return projector and correction options for each method.

    The five entries are explicit: one mapping, consumed by one reconstruction
    loop. ``reference_normaliser`` takes its component first and its ``method``
    as a keyword argument.
    """

    return {
        "Known activity": {
            "projector": {
                "activity_mbq": lambda image: activity_mbq,
                "time_per_projection_s": time_per_projection_s,
            },
            "correction": {},
        },
        "Reference, sum": {
            "projector": {"normalise": reference_normaliser(full_model, method="sum")},
            "correction": {},
        },
        "Reference, trimmed": {
            "projector": {
                "normalise": reference_normaliser(full_model, method="trimmed")
            },
            "correction": {},
        },
        "Dewaraja, global": {
            "projector": {"normalise": measured_normaliser(measured)},
            "correction": {},
        },
        "Dewaraja, pixel-wise": {
            "projector": {"normalise": measured_normaliser(measured)},
            "correction": {"measured": measured},
        },
    }
