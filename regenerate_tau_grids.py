"""Regenerate Duffy/Uchuu MAT grids; --extend-to-z3 writes distinct z=3..100 grids."""
import argparse
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import shutil
import numpy as np
from scipy.io import savemat, loadmat
from colossus import settings

# Keep Colossus caches inside the writable project directory.
ROOT = Path(__file__).resolve().parent
settings.BASE_DIR = str(ROOT / 'outputs' / 'colossus_cache')

import halo_tau_calculator_vectorised as calculator
from halo_tau_integrator import get_tau_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extend-to-z3', action='store_true',
                        help='Use z=3..100 and write *_z3.mat instead of the original grids.')
    args = parser.parse_args()
    redshifts = (np.arange(3, 101, 1) if args.extend_to_z3
                 else calculator.redshift_bins_wide)
    suffix = '_z3' if args.extend_to_z3 else ''
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = ROOT / 'outputs' / 'grid_backups' / stamp
    backup.mkdir(parents=True)
    shape = (len(calculator.coarse_freqs), len(calculator.xHI_bins),
             len(calculator.halo_masses_Msun), len(redshifts))
    for model, filename in [('duffy', f'halo_tau_grid{suffix}.mat'),
                            ('uchuu', f'halo_tau_grid_uchuu{suffix}.mat')]:
        target = ROOT / filename
        if target.exists():
            shutil.copy2(target, backup / filename)
        tau = np.empty(shape)
        for index, mass in enumerate(calculator.halo_masses_Msun):
            with redirect_stdout(io.StringIO()):
                integrand, radii = calculator.get_integrand_grid(
                    calculator.coarse_freqs, redshifts,
                    np.array([mass]), calculator.xHI_bins,
                    concentration_model=model)
                tau[:, :, index, :] = get_tau_grid(integrand, radii).reshape(
                    shape[0], shape[1], shape[3])
            if (index+1) % 10 == 0:
                print(f'{model}: {index+1}/{shape[2]} masses', flush=True)
        assert np.all(np.isfinite(tau)) and np.all(tau >= 0)
        payload = dict(tau_arr=tau, freq_bins=calculator.coarse_freqs,
                       redshift_bins=redshifts,
                       halo_masses_Msun=calculator.halo_masses_Msun,
                       xHI_bins=calculator.xHI_bins,
                       concentration_model=model, radial_samples=100,
                       inner_radial_samples=100, helium_extra_divisors_removed=1,
                       log10_radial_jacobian_corrected=1)
        temporary = target.with_suffix('.tmp.mat')
        savemat(temporary, payload)
        saved = loadmat(temporary)
        np.testing.assert_array_equal(saved['tau_arr'], tau)
        for key in ['freq_bins','redshift_bins','halo_masses_Msun','xHI_bins']:
            np.testing.assert_array_equal(saved[key].ravel(), payload[key])
        assert saved['tau_arr'].shape == shape
        temporary.replace(target)
        print(f'Saved {filename}: shape={shape}, tau range={tau.min():.6g}..{tau.max():.6g}', flush=True)
    print(f'Previous grids backed up in {backup}', flush=True)


if __name__ == '__main__':
    main()
