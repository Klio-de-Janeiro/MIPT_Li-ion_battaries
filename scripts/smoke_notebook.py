"""Execute the actual notebook on clearly synthetic CSVs in an isolated directory."""
from __future__ import annotations

import argparse
import contextlib
import io
import traceback
import os
import shutil
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import polars as pl
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager


def make_synthetic_logs(root):
    root.mkdir(parents=True, exist_ok=True)
    for cell in range(10):
        interval = 2 if cell < 2 else 30
        n = 120 * (30 // interval)
        time = np.arange(n, dtype=float) * interval
        capacity = np.full(n, np.nan)
        capacity[0] = 3 - .02*cell
        capacity[-1] = 2.9 - .02*cell
        pl.DataFrame(dict(timestamp_s=time, EFC=time/12000,
                          v_raw_V=3.6+.3*np.sin(time/700)+cell*.01,
                          i_raw_A=np.cos(time/700), t_cell_degC=25+np.sin(time/900),
                          soc_est=50+30*np.sin(time/700), cap_aged_est_Ah=capacity)).write_csv(
            root/f'cell_log_age_{interval}s_P{cell:03d}_1_S01_C01.csv', separator=';')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--in-process', action='store_true', help='Execute notebook cells without a Jupyter socket/kernel')
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]
    work=(args.work_dir or Path(tempfile.mkdtemp(prefix='mden_smoke_'))).resolve()
    if work == source or source in work.parents:
        raise ValueError('Place smoke work directory outside the source project')
    work.mkdir(parents=True,exist_ok=True)
    project=work/'project'
    def ignore(folder, names):
        excluded = {'__pycache__', '.pytest_cache'}
        if Path(folder).resolve() == source:
            excluded.update({'data', 'runs'})
        return [name for name in names if name in excluded]
    shutil.copytree(source,project,ignore=ignore,dirs_exist_ok=True)
    make_synthetic_logs(work/'raw')
    os.environ['MDEN_PROJECT_ROOT']=str(project)
    os.environ['MDEN_RAW_DATA_DIR']=str(work/'raw')
    os.environ['MDEN_SMOKE_TEST']='1'
    os.environ['MPLBACKEND']='Agg'
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    notebook=nbformat.read(project/'04_train_v2_fixed.ipynb',as_version=4)
    try:
        if args.in_process:
            namespace={'__name__':'__main__'}
            os.chdir(project)
            executed=0
            for index,cell in enumerate(notebook.cells):
                if cell.cell_type != 'code':
                    continue
                executed += 1
                cell.execution_count=executed
                stream=io.StringIO()
                try:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        exec(compile(cell.source,f'notebook_cell_{index}','exec'),namespace)
                finally:
                    cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue())]
                    print(f'cell {index}: {stream.getvalue()[-500:]}',flush=True)
            print(f'Executed {executed} code cells in one Python process (no kernel UI test)')
        else:
            NotebookClient(notebook,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(project)}}).execute()
    finally:
        nbformat.write(notebook,work/'executed_smoke.ipynb')
    print('PASS: actual notebook executed on synthetic data; outputs:',work)


if __name__=='__main__':
    main()
