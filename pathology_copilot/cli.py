"""Typer CLI: `pathology-copilot run --case demo/cases/demo1.json`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .orchestrator import Orchestrator
from .schemas import CriticOutput, ToolResult
from .state import ExternalState


app = typer.Typer(add_completion=False, help="Pathology AI Copilot demo CLI")
console = Console()


@app.callback()
def _root() -> None:
    """Root callback keeps `run` as a subcommand instead of collapsing to the app."""


def _print_step(state: ExternalState, result: ToolResult) -> None:
    status = "OK" if result.error is None else f"ERR({result.error})"
    region = result.grounding[0].region_id if result.grounding else "-"
    console.print(
        f"[cyan]\\[Executor][/cyan] step {state.step}/{len(state.plan)}: "
        f"[bold]{result.tool_name}[/bold] on {region} "
        f"uncertainty={result.uncertainty:.2f} {status}"
    )


def _print_critic(round_num: int, output: CriticOutput) -> None:
    if not output.notes:
        console.print(
            f"[magenta]\\[Self-Critic] round {round_num}: 0 flags — passed[/magenta]"
        )
        return
    console.print(
        f"[magenta]\\[Self-Critic] round {round_num}: "
        f"{len(output.notes)} flag(s)[/magenta]"
    )
    for note in output.notes:
        console.print(f"  - [{note.kind}] {note.message}")


def _print_report(state: ExternalState) -> None:
    table = Table(title="Diagnostic Report", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")


@app.command()
def run(
    case: Path = typer.Option(..., exists=True, help="Path to case JSON."),
    dump_state: Optional[Path] = typer.Option(
        None, help="Write state JSON here after the run (or after --stop-after)."
    ),
    stop_after: Optional[int] = typer.Option(
        None, help="Stop after N executor steps (for demo-ing resume)."
    ),
    resume: Optional[Path] = typer.Option(
        None, exists=True, help="Resume from a state JSON file."
    ),
) -> None:
    """Run the pathology copilot on a case."""

    case_data = json.loads(case.read_text())
    case_id = case_data["case_id"]

    state: Optional[ExternalState] = None
    if resume is not None:
        state = ExternalState.load(resume)
        case_id = state.case_id
        console.print(
            Panel.fit(
                f"Resuming case [bold]{state.case_id}[/bold] at step {state.step}",
                style="yellow",
            )
        )
    else:
        console.print(
            Panel.fit(
                f"Case [bold]{case_id}[/bold]: {case_data.get('clinical_info', '')}",
                style="green",
            )
        )

    orchestrator = Orchestrator(step_hook=_print_step, critic_hook=_print_critic)
    state, report = orchestrator.run(
        case_id=case_id,
        case_metadata=case_data,
        state=state,
        max_steps=stop_after,
    )

    if not state.finished:
        console.print(
            "[yellow]Stopped early. Use --resume to continue.[/yellow]"
        )
        if dump_state is not None:
            state.dump(dump_state)
            console.print(f"State written to {dump_state}")
        else:
            fallback = Path(f"state_{state.case_id}.json")
            state.dump(fallback)
            console.print(f"State written to {fallback}")
        return

    console.print()
    assert report is not None
    table = Table(title="Diagnostic Report", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("case_id", report.case_id)
    table.add_row("primary_diagnosis", report.primary_diagnosis)
    table.add_row("subtype", str(report.subtype))
    table.add_row("grade", str(report.grade))
    table.add_row("mutations", ", ".join(report.mutations) or "-")
    table.add_row("confidence", f"{report.confidence:.2f}")
    table.add_row("findings", str(len(report.findings)))
    console.print(table)

    console.print("\n[bold]Findings:[/bold]")
    for ev in report.findings:
        regions = ", ".join(g.region_id for g in ev.grounding) or "-"
        console.print(
            f"  - {ev.statement}  "
            f"[dim](conf={ev.confidence:.2f}, grounding={{{regions}}})[/dim]"
        )

    if dump_state is not None:
        state.dump(dump_state)
        console.print(f"\nState written to {dump_state}")


if __name__ == "__main__":  # pragma: no cover
    app()
