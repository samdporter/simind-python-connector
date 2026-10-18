#!/usr/bin/env python
"""Simulate a known-activity NEMA phantom and reconstruct it in STIR or SIRF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


try:
    from _python_connector_helpers import require_simind
except ModuleNotFoundError:
    from examples._python_connector_helpers import require_simind

from simind_python_connector import configs
from simind_python_connector.builders import STIRSPECTImageDataBuilder
from simind_python_connector.converters.attenuation import get_attenuation_coefficient
from simind_python_connector.core.types import DEFAULT_QUANTIZATION_SCALE
from simind_python_connector.phantoms import nema_iec_phantom


VOXELS = 96
VOXEL_MM = 4.0


def _write_image(array: np.ndarray, base: Path) -> Path:
    builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(array.shape[2]),
            "!matrix size [2]": str(array.shape[1]),
            "!matrix size [3]": str(array.shape[0]),
            "scaling factor (mm/pixel) [1]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [2]": str(VOXEL_MM),
            "scaling factor (mm/pixel) [3]": str(VOXEL_MM),
        }
    )
    builder.set_pixel_array(array.astype(np.float32))
    builder.build(output_path=base)
    return base.with_suffix(".hv")


def _reconstruct(backend: str, projection_path: Path, mu_path: Path):
    if backend == "sirf":
        import sirf.STIR as sirf

        projection = sirf.AcquisitionData(str(projection_path))
        mu = sirf.ImageData(str(mu_path))
        initial = sirf.ImageData(str(mu_path)).get_uniform_copy(1.0)
        # STIR SPECTUB mirrors x relative to the mu map (UCL/STIR#452)
        mu.fill(mu.as_array()[..., ::-1])
        matrix = sirf.SPECTUBMatrix()
        matrix.set_attenuation_image(mu)
        model = sirf.AcquisitionModelUsingMatrix(matrix)
        objective = sirf.make_Poisson_loglikelihood(projection)
        objective.set_acquisition_model(model)
        objective.set_num_subsets(2)
        recon = sirf.OSMAPOSLReconstructor()
        recon.set_objective_function(objective)
        recon.set_num_subsets(2)
        recon.set_num_subiterations(2)
        recon.set_input(projection)
        recon.set_up(initial)
        recon.reconstruct(initial)
        recon_path = Path("output/nema_reconstruction") / "sirf_recon.hv"
        initial.write(str(recon_path))
        projection_array = projection.as_array()
        reconstruction_array = initial.as_array()
        print(f"Projection shape: {projection_array.shape}")
        print(f"Reconstruction shape: {reconstruction_array.shape}")
        print(f"Reconstruction finite: {np.isfinite(reconstruction_array).all()}")
        print(f"Reconstruction sum: {reconstruction_array.sum():.4g}")
    else:
        import stir

        from simind_python_connector.utils import get_array

        projection = stir.ProjData.read_from_file(str(projection_path))
        mu = stir.FloatVoxelsOnCartesianGrid.read_from_file(str(mu_path))
        initial = mu.get_empty_copy()
        initial.fill(1.0)
        # STIR SPECTUB mirrors x relative to the mu map (UCL/STIR#452)
        mu.fill(get_array(mu)[..., ::-1].flat)
        matrix = stir.ProjMatrixByBinSPECTUB()
        matrix.set_attenuation_image_sptr(mu)
        pair = stir.ProjectorByBinPairUsingProjMatrixByBin()
        pair.set_proj_matrix_sptr(matrix)
        objective = stir.PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat()
        objective.set_projector_pair_sptr(pair)
        objective.set_proj_data_sptr(projection)
        objective.set_num_subsets(2)
        recon = stir.OSMAPOSLReconstruction3DFloat()
        recon.set_objective_function(objective)
        recon.set_num_subsets(2)
        recon.set_num_subiterations(2)
        recon.set_disable_output(True)
        recon.set_up(initial)
        recon.reconstruct(initial)

        recon_path = Path("output/nema_reconstruction") / "stir_recon.hv"
        initial.write_to_file(str(recon_path))
        projection_array = get_array(projection)
        reconstruction_array = get_array(initial)
        print(f"Projection shape: {projection_array.shape}")
        print(f"Reconstruction shape: {reconstruction_array.shape}")
        print(f"Reconstruction finite: {np.isfinite(reconstruction_array).all()}")
        print(f"Reconstruction sum: {reconstruction_array.sum():.4g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("stir", "sirf"), required=True)
    parser.add_argument(
        "--simind-version",
        type=int,
        choices=(8, 9),
        default=None,
        help="SIMIND major version (defaults to SIMIND_VERSION, then 8)",
    )
    args = parser.parse_args()
    backend = args.backend
    require_simind()
    output = Path("output/nema_reconstruction")
    output.mkdir(parents=True, exist_ok=True)

    phantom = nema_iec_phantom((VOXELS, VOXELS, VOXELS), (VOXEL_MM,) * 3, preset="pet")
    source_path = _write_image(phantom.activity_mbq, output / "activity")
    mu_water = get_attenuation_coefficient("water", 140.0)
    mu_path = _write_image(phantom.density_g_cm3 * mu_water, output / "mu")
    np.save(output / "activity_mbq_per_voxel.npy", phantom.activity_mbq)
    np.save(output / "density_g_cm3.npy", phantom.density_g_cm3)
    print(f"NEMA source shape: {phantom.activity_mbq.shape}")

    if backend == "sirf":
        from simind_python_connector import SirfSimindAdaptor as Adaptor
    else:
        from simind_python_connector import StirSimindAdaptor as Adaptor
    adaptor = Adaptor(
        configs.get("Example.yaml"),
        output,
        f"nema_{backend}",
        photon_multiplier=1,
        quantization_scale=0.01 * DEFAULT_QUANTIZATION_SCALE,
        simind_version=args.simind_version,
    )
    if backend == "sirf":
        import sirf.STIR as sirf

        source, mu = sirf.ImageData(str(source_path)), sirf.ImageData(str(mu_path))
    else:
        import stir

        source = stir.FloatVoxelsOnCartesianGrid.read_from_file(str(source_path))
        mu = stir.FloatVoxelsOnCartesianGrid.read_from_file(str(mu_path))
    adaptor.set_source(source)
    adaptor.set_mu_map(mu)
    adaptor.set_energy_windows([126], [154], [0])
    adaptor.add_runtime_switch("FI", "tc99m")
    adaptor.add_runtime_switch("CC", "ma-lehr")
    adaptor.add_runtime_switch("NN", 1)
    adaptor.add_runtime_switch("RR", 12345)
    adaptor.add_config_value(1, 140.0)
    adaptor.add_config_value(29, 4)
    adaptor.add_config_value(28, VOXEL_MM / 10.0)
    adaptor.add_config_value(76, VOXELS)
    adaptor.add_config_value(77, VOXELS)
    adaptor.set_activity(float(phantom.activity_mbq.sum()), time_per_projection_s=1.0)
    adaptor.run()
    projection = adaptor.get_total_output(window=1)
    projection_path = output / f"{backend}_total.hs"
    if backend == "sirf":
        projection.write(str(projection_path))
    else:
        projection.write_to_file(str(projection_path))
    print(f"NEMA activity: {phantom.activity_mbq.sum():.4g} MBq")
    print(f"Projection sum: {projection.sum():.4g}")
    _reconstruct(backend, projection_path, mu_path)
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
