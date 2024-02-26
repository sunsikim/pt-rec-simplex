import typer
import jobs
from jobs.config import ExecutableJobs
from typing import Annotated
from typer import Typer

app = Typer()


@app.command("list-jobs")
def list_jobs():
    for job in ExecutableJobs.__subclasses__():
        _job = job()
        print(_job.name, _job.__doc__)


@app.command("run-jobs")
def run_jobs(jobs: Annotated[str, typer.Option(envvar="APP_JOBS", help="comma-separated sequence of job commands")]):
    # register executable jobs
    executable_jobs = {}
    for job in ExecutableJobs.__subclasses__():
        _job = job()
        executable_jobs[_job.name] = _job.execute

    # execute input jobs
    for input_job in jobs.split(","):
        if input_job not in executable_jobs:
            raise ValueError(f"'{input_job}' is not one of registered jobs {tuple(executable_jobs.keys())}")
        else:
            executable_jobs[input_job]()
